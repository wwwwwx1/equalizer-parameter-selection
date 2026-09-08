Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
shell.CurrentDirectory = files.GetParentFolderName(WScript.ScriptFullName)
pythonExe = ""
configFile = files.BuildPath(shell.CurrentDirectory, "python_path.txt")
If files.FileExists(configFile) Then
    Set stream = CreateObject("ADODB.Stream")
    stream.Type = 2
    stream.Charset = "utf-8"
    stream.Open
    stream.LoadFromFile configFile
    pythonExe = Trim(stream.ReadText)
    stream.Close
End If
If pythonExe = "" Then pythonExe = files.BuildPath(shell.CurrentDirectory, ".venv\Scripts\python.exe")
If Not files.FileExists(pythonExe) Then
    MsgBox "Set your existing python.exe path in python_path.txt first.", 16, "Python setup"
    WScript.Quit 1
End If
shell.Run Chr(34) & pythonExe & Chr(34) & " -X utf8 -m scripts.workbench_gui", 0, False
