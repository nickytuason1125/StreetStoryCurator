$ProgressPreference = 'SilentlyContinue'
$log = 'C:\Users\Nicky Tuason\Desktop\StreetPhotoEditor\street-story-curator\model_download.log'
't pull start ' + (Get-Date) | Out-File $log -Encoding utf8
try {
  $req = [System.Net.HttpWebRequest]::Create('http://127.0.0.1:8000/api/models/pull')
  $req.Method = 'POST'; $req.ContentType = 'application/json'
  $req.Timeout = 60000; $req.ReadWriteTimeout = 600000
  $b = [Text.Encoding]::UTF8.GetBytes('{"model_name":"optional"}')
  $req.ContentLength = $b.Length
  $rs = $req.GetRequestStream(); $rs.Write($b,0,$b.Length); $rs.Close()
  $resp = $req.GetResponse()
  $sr = New-Object IO.StreamReader($resp.GetResponseStream())
  while (-not $sr.EndOfStream) {
    $line = $sr.ReadLine()
    if ($line) { ((Get-Date -Format 'HH:mm:ss') + ' ' + $line) | Out-File $log -Append -Encoding utf8 }
  }
  $resp.Close()
  'STREAM COMPLETE' | Out-File $log -Append -Encoding utf8
} catch { ('ERR: ' + $_.Exception.Message) | Out-File $log -Append -Encoding utf8 }
