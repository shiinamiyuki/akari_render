$CUR_DIR = Get-Location
$env:Path += ";$CUR_DIR/build/dist/"
$env:PYTHONPATH += "$CUR_DIR/build/dist/"