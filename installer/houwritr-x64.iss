; Inno Setup script — Intel/AMD (x86_64) build of HouWritr.
;
; CPU-only: Parakeet runs on the ONNX Runtime CPU EP. No Qualcomm QNN
; runtime (Hexagon is Snapdragon-only). Much smaller than the arm64 build.
;
; Build:
;   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" /Qp installer\houwritr-x64.iss

#define AppName       "HouWritr"
#ifndef AppVersion
#define AppVersion    "0.6.1"
#endif
#define AppPublisher  "Wenxuan Hou"
#define AppURL        "https://github.com/raulhou/HouWritr"
#define AppExeName    "houwritr.exe"
#define SrcDir        "..\target\stage\x64"

[Setup]
AppId={{C105EA40-33DF-4139-9139-02B5C5650D19}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DefaultDirName={localappdata}\HouWritr\app
DefaultGroupName={#AppName}
DisableProgramGroupPage=auto
; x64 installer: runs on Intel/AMD Windows. (Also runs under x64 emulation on
; ARM64, but those users want the native arm64 build.)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\target\dist
OutputBaseFilename=houwritr-windows-x64-v{#AppVersion}-setup
Compression=lzma2/ultra
SolidCompression=yes
WizardStyle=modern
CloseApplications=force
RestartApplications=no
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
SetupIconFile=houwritr.ico
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
LicenseFile=..\LICENSE

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"

[Tasks]
Name: "autostart"; Description: "Start {#AppName} automatically when I log in"; GroupDescription: "Startup:"
Name: "startmenuicon"; Description: "Create a Start Menu shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
; The canonical release stage contains no Qualcomm-only files on x64.
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: startmenuicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Autostart is a HKCU\...\Run value so the app can read, toggle, and verify the
; exact same mechanism at runtime (Settings → Startup). uninsdeletevalue removes
; it on uninstall even if the task was unselected but later enabled in the app.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "HouWritr"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart
; The startup value had another name before the rename; the autostart task is
; preselected below when it was there, so nothing is lost by removing it.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: none; ValueName: "OpenWritr"; Flags: deletevalue

[InstallDelete]
; Migrate away from the legacy startup-folder shortcut used by older installers,
; so autostart state never lives in two mechanisms at once. Gated on the autostart
; task so we only remove the shortcut when we actually write the Run value in its
; place. If the user deselects autostart on upgrade, the shortcut is left untouched
; and the app-side migration in autostart.rs remains able to pick it up later —
; a previously active autostart is never silently dropped.
Type: files; Name: "{userstartup}\{#AppName}.lnk"; Tasks: autostart
; The program file had another name before the rename; an upgrade installs
; into the same folder, so remove the old one.
Type: files; Name: "{app}\openwritr.exe"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{cmd}"; Parameters: "/C taskkill /IM {#AppExeName} /F"; Flags: runhidden; RunOnceId: "KillHouWritr"
; Always remove the autostart Run value, even if the user enabled it from inside
; the app after installing without the autostart task (so uninsdeletevalue never
; recorded it). reg delete is a no-op with exit 1 if it is already gone.
Filename: "{cmd}"; Parameters: "/C reg delete ""HKCU\Software\Microsoft\Windows\CurrentVersion\Run"" /v HouWritr /f"; Flags: runhidden; RunOnceId: "RemoveHouWritrRun"

[UninstallDelete]
; Leave user data (settings, models, logs) under %LOCALAPPDATA%\HouWritr\ alone.
Type: files; Name: "{userstartup}\{#AppName}.lnk"
Type: filesandordirs; Name: "{app}"

[Code]
// Preselect the autostart task on upgrade when this user already had autostart
// enabled by an older install — either the legacy startup-folder shortcut or the
// HKCU Run value. Inno remembers the previous task selection across upgrades, so
// a user who once deselected the task would otherwise get it deselected again by
// default; this restores it to checked when prior autostart state is detected.
function LegacyAutostartEnabled(): Boolean;
begin
  Result := FileExists(ExpandConstant('{userstartup}\{#AppName}.lnk')) or
    RegValueExists(HKEY_CURRENT_USER,
      'Software\Microsoft\Windows\CurrentVersion\Run', 'HouWritr') or
    RegValueExists(HKEY_CURRENT_USER,
      'Software\Microsoft\Windows\CurrentVersion\Run', 'OpenWritr');
end;

procedure InitializeWizard();
begin
  if LegacyAutostartEnabled() then
    WizardSelectTasks('autostart');
end;

// The program had another name before the rename: stop a copy of the old one
// that is still running before its file is removed.
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
    Exec(ExpandConstant('{cmd}'), '/C taskkill /IM openwritr.exe /F', '',
      SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
