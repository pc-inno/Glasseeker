#!/usr/bin/env python3
"""Internal launcher for execute_code's host-filesystem-denied mode.

The launcher runs inside fresh Linux user/mount/PID/IPC/UTS namespaces. It
replaces every non-runtime top-level directory with an empty tmpfs, copies the
per-call script and RPC module into a private tmpfs, makes that tmpfs read-only,
installs a seccomp filter that rejects filesystem mutations, drops capabilities,
and finally execs the system Python interpreter.

It is intentionally stdlib-only and fail-closed. Any setup error exits before
model-authored code runs.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import subprocess
import sys
from pathlib import Path


MOUNT_BIN = "/usr/bin/mount"
SYSTEM_PYTHON = "/usr/bin/python3"
SANDBOX_DIR = Path("/tmp/hermes_exec")

# Keep only the immutable OS/Python runtime hierarchy visible. Conventional
# merged-/usr entries are symlinks into /usr on supported hosts.
RUNTIME_TOP_LEVEL = frozenset({"usr", "bin", "sbin", "lib", "lib64", "lib32", "libx32"})

# libseccomp constants from seccomp.h.
SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO_BASE = 0x00050000
SCMP_CMP_MASKED_EQ = 7


class ScmpArgCmp(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _run_mount(*args: str) -> None:
    command = [MOUNT_BIN, "-n", *args]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")


def _mount_empty_read_only(path: Path) -> None:
    _run_mount(
        "-t", "tmpfs",
        "-o", "size=4k,mode=0555,ro,nosuid,nodev,noexec",
        "tmpfs", str(path),
    )


def _load_staging(staging: Path) -> tuple[bytes, bytes]:
    return (
        (staging / "script.py").read_bytes(),
        (staging / "hermes_tools.py").read_bytes(),
    )


def _prepare_private_tmp(script: bytes, hermes_tools: bytes) -> None:
    _run_mount(
        "-t", "tmpfs",
        "-o", "size=32m,mode=0755,nosuid,nodev,noexec",
        "tmpfs", "/tmp",
    )
    SANDBOX_DIR.mkdir(mode=0o755)
    script_path = SANDBOX_DIR / "script.py"
    tools_path = SANDBOX_DIR / "hermes_tools.py"
    script_path.write_bytes(script)
    tools_path.write_bytes(hermes_tools)
    script_path.chmod(0o444)
    tools_path.chmod(0o444)
    SANDBOX_DIR.chmod(0o555)
    _run_mount("-o", "remount,ro,nosuid,nodev,noexec", "/tmp")


def _seccomp_library():
    seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(ScmpArgCmp),
    ]
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    return seccomp


def _errno_action(value: int) -> int:
    return SCMP_ACT_ERRNO_BASE | (value & 0xFFFF)


def _add_deny_rule(seccomp, context, syscall: str, comparisons=()) -> None:
    number = seccomp.seccomp_syscall_resolve_name(syscall.encode("ascii"))
    if number < 0:  # syscall absent on this architecture/kernel headers
        return
    array = (ScmpArgCmp * len(comparisons))(*comparisons) if comparisons else None
    result = seccomp.seccomp_rule_add_array(
        context,
        _errno_action(errno.EACCES),
        number,
        len(comparisons),
        array,
    )
    if result != 0:
        raise RuntimeError(f"seccomp rule for {syscall} failed: errno {-result}")


def _masked(argument: int, mask: int, value: int) -> ScmpArgCmp:
    return ScmpArgCmp(argument, SCMP_CMP_MASKED_EQ, mask, value)


def _install_filesystem_seccomp() -> None:
    seccomp = _seccomp_library()
    context = seccomp.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("seccomp_init returned null")

    try:
        # open(path, flags, ...) / openat(dirfd, path, flags, ...): reject
        # every write-capable access mode and creation/truncation flag while
        # leaving read-only imports and runtime reads intact.
        access_mode = os.O_WRONLY | os.O_RDWR
        write_bits = (
            os.O_CREAT,
            os.O_TRUNC,
            os.O_APPEND,
            getattr(os, "O_TMPFILE", 0),
        )
        for syscall, flag_argument in (("open", 1), ("openat", 2)):
            _add_deny_rule(
                seccomp,
                context,
                syscall,
                (_masked(flag_argument, access_mode, os.O_WRONLY),),
            )
            _add_deny_rule(
                seccomp,
                context,
                syscall,
                (_masked(flag_argument, access_mode, os.O_RDWR),),
            )
            for bit in write_bits:
                if bit:
                    _add_deny_rule(
                        seccomp,
                        context,
                        syscall,
                        (_masked(flag_argument, bit, bit),),
                    )

        # openat2 stores flags behind a pointer, which classic seccomp cannot
        # inspect safely. Deny it outright; glibc/Python use openat here.
        always_deny = (
            "openat2", "creat", "truncate", "ftruncate", "fallocate",
            "unlink", "unlinkat", "rename", "renameat", "renameat2",
            "mkdir", "mkdirat", "rmdir", "link", "linkat", "symlink",
            "symlinkat", "mknod", "mknodat", "chmod", "fchmod",
            "fchmodat", "chown", "fchown", "fchownat", "lchown",
            "utime", "utimes", "futimesat", "utimensat", "setxattr",
            "lsetxattr", "fsetxattr", "removexattr", "lremovexattr",
            "fremovexattr", "mount", "umount2", "pivot_root", "chroot",
            "swapon", "swapoff", "quotactl", "open_by_handle_at",
            "name_to_handle_at", "open_tree", "move_mount", "fsopen",
            "fsconfig", "fsmount", "fspick", "mount_setattr", "unshare",
            "setns", "io_uring_setup", "io_uring_enter", "io_uring_register",
            "ptrace", "process_vm_writev",
        )
        for syscall in always_deny:
            _add_deny_rule(seccomp, context, syscall)

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"seccomp_load failed: errno {-result}")
    finally:
        seccomp.seccomp_release(context)


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)

    # Drop every capability from the bounding set while CAP_SETPCAP is still
    # effective. EINVAL means the kernel does not define that capability.
    for capability in range(64):
        if libc.prctl(24, capability, 0, 0, 0) != 0:  # PR_CAPBSET_DROP
            error = ctypes.get_errno()
            if error != errno.EINVAL:
                raise OSError(error, f"PR_CAPBSET_DROP({capability}) failed")

    # Prevent uid 0 from regaining capabilities across execve.
    securebits = 0x1 | 0x2 | 0x4 | 0x8  # NOROOT(+lock), NO_SETUID_FIXUP(+lock)
    if libc.prctl(28, securebits, 0, 0, 0) != 0:  # PR_SET_SECUREBITS
        raise OSError(ctypes.get_errno(), "PR_SET_SECUREBITS failed")

    header = CapHeader(0x20080522, 0)  # _LINUX_CAPABILITY_VERSION_3
    data = (CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "capset failed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--staging", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    staging = Path(args.staging).resolve()

    try:
        script, hermes_tools = _load_staging(staging)
        top_level_directories = [
            path
            for path in Path("/").iterdir()
            if path.is_dir() and path.name not in RUNTIME_TOP_LEVEL
        ]

        # /tmp needs a private copy of the generated code; all other data
        # directories become empty and read-only.
        for path in top_level_directories:
            if path == Path("/tmp"):
                continue
            _mount_empty_read_only(path)
        _prepare_private_tmp(script, hermes_tools)
        os.chdir(SANDBOX_DIR)

        _install_filesystem_seccomp()
        _drop_capabilities()

        child_env = dict(os.environ)
        child_env["HOME"] = "/nonexistent"
        child_env["TMPDIR"] = "/tmp"
        child_env["TMP"] = "/tmp"
        child_env["TEMP"] = "/tmp"
        child_env["PYTHONPATH"] = str(SANDBOX_DIR)
        child_env["PATH"] = "/usr/bin:/bin"
        for host_path_variable in (
            "VIRTUAL_ENV",
            "CONDA_PREFIX",
            "PYTHONHOME",
            "HERMES_HOME",
            "HERMES_CONFIG",
            "HERMES_ENV",
        ):
            child_env.pop(host_path_variable, None)

        os.execve(
            SYSTEM_PYTHON,
            [SYSTEM_PYTHON, str(SANDBOX_DIR / "script.py")],
            child_env,
        )
    except Exception as exc:
        print(
            f"execute_code filesystem isolation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 126


if __name__ == "__main__":
    raise SystemExit(main())
