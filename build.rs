//! Build script: embed the app icon + version metadata into houwritr.exe
//! on Windows. The icon lives at installer/houwritr.ico (regenerate with
//! `python installer/make_icon.py`). On non-Windows targets this is a no-op
//! so `cargo check` still works on a dev box.

fn main() {
    #[cfg(target_os = "windows")]
    {
        // Only the GUI binary needs the icon; the `package` helper bin doesn't,
        // but winresource attaches per-crate so both get it — harmless.
        let mut res = winresource::WindowsResource::new();
        res.set_icon("installer/houwritr.ico");
        res.set("ProductName", "HouWritr");
        res.set(
            "FileDescription",
            "HouWritr — meeting recording and file transcription",
        );
        res.set("CompanyName", "Wenxuan Hou");
        res.set(
            "LegalCopyright",
            "Copyright (c) 2026 Wenxuan Hou. Based on OpenWritr, copyright (c) 2026 Torsten Mahr and contributors (MIT)",
        );
        if let Err(e) = res.compile() {
            // Don't hard-fail the build if the resource compiler isn't on PATH
            // (e.g. a shell without vcvars). Just warn — the exe builds, sans icon.
            println!("cargo:warning=icon embed skipped: {e}");
        }
        println!("cargo:rerun-if-changed=installer/houwritr.ico");
    }
}
