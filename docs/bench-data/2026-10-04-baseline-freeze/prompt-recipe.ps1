param(
    [Parameter(Mandatory = $true)]
    [string] $OutputPath
)

$lines = @(
    'This is a deterministic long-context benchmark record. It contains source code,',
    'configuration values, function names, variables, numbers, documentation, and prose.',
    'The runtime must preserve earlier information while processing later sections.',
    'Function process_request validates input, updates state, reads cached values,',
    'performs attention, executes routed experts, and returns a structured result.',
    'Values: alpha=17 beta=291 gamma=4096 delta=65535.'
)
$block = $lines -join "`n"
$suffix = @(
    '',
    'Continue now with exactly 64 short numbered lines.',
    'Each line must have the form:',
    'LINE <number>: benchmark continuation text',
    'Start at LINE 1 and do not stop before LINE 64.'
) -join "`n"
$prompt = (($block + "`n") * 668) + $suffix
$bytes = [System.Text.Encoding]::UTF8.GetBytes($prompt)
[System.IO.File]::WriteAllBytes($OutputPath, $bytes)
$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    $digest = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '').ToLowerInvariant()
    Write-Output "bytes=$($bytes.Length) sha256=$digest"
} finally {
    $sha.Dispose()
}
