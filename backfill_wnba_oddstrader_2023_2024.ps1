param(
    [string]$RepoRoot = "",
    [int]$DelayMilliseconds = 300
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# OddsTrader internal structured odds endpoint.
$ApiEndpoint = "https://www.oddstrader.com/odds-v2/odds-v2-service"

# Confirmed by the open-source sbrscrape implementation for basketball:
# 401 = spread, 402 = total, 83 = moneyline.
$MarketIds = @(401, 402, 83)

# Provider IDs used by OddsTrader/SBR. DraftKings = 91.
$ProviderNames = @{
    3   = "Bookmaker"
    5   = "WilliamHill"
    8   = "BetOnline"
    9   = "SportsBetting"
    10  = "Pinnacle"
    15  = "MyBookie"
    16  = "GTBets"
    18  = "888Sport"
    20  = "BetAnySports"
    22  = "SugarHouse"
    28  = "Intertops"
    29  = "LowVig"
    35  = "BetCRIS"
    36  = "Unibet"
    38  = "JustBet"
    44  = "Heritage"
    45  = "Wynn"
    54  = "Resorts"
    65  = "BetUS"
    78  = "FanDuel"
    82  = "Bodog"
    83  = "BetNow"
    84  = "Bovada"
    91  = "DraftKings"
    92  = "EveryGame"
    123 = "Consensus"
}
$ProviderIds = @($ProviderNames.Keys | Sort-Object)

function Resolve-RepoRoot {
    param([string]$Requested)

    if ($Requested) {
        $candidate = (Resolve-Path $Requested).Path
        if (Test-Path (Join-Path $candidate "docs\win\basketball")) {
            return $candidate
        }
        throw "RepoRoot does not contain docs\win\basketball: $candidate"
    }

    $candidates = @()
    if ($PSScriptRoot) { $candidates += $PSScriptRoot }
    $candidates += (Get-Location).Path

    foreach ($start in $candidates) {
        $dir = [System.IO.DirectoryInfo]::new($start)
        while ($null -ne $dir) {
            if (Test-Path (Join-Path $dir.FullName "docs\win\basketball")) {
                return $dir.FullName
            }
            $dir = $dir.Parent
        }
    }

    throw "Could not locate repo root containing docs\win\basketball."
}

function Convert-AmericanToDecimal {
    param($American)

    if ($null -eq $American -or "$American".Trim() -eq "") { return $null }

    $a = 0.0
    if (-not [double]::TryParse("$American", [ref]$a)) { return $null }
    if ($a -eq 0) { return $null }

    if ($a -gt 0) {
        return [math]::Round(1.0 + ($a / 100.0), 4)
    }

    return [math]::Round(1.0 + (100.0 / [math]::Abs($a)), 4)
}

function Normalize-TeamForId {
    param([string]$Name)
    if (-not $Name) { return "Unknown" }

    $s = $Name.Trim()
    $s = [regex]::Replace($s, "[^\p{L}\p{Nd}]+", "_")
    $s = $s.Trim("_")
    return $s
}

function Get-ProviderName {
    param([int]$Paid)
    if ($ProviderNames.ContainsKey($Paid)) {
        return $ProviderNames[$Paid]
    }
    return "Provider_$Paid"
}

$Headers = @{
    "User-Agent"      = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131 Safari/537.36"
    "Accept"          = "application/json,text/plain,*/*"
    "Accept-Language" = "en-US,en;q=0.9"
    "Referer"         = "https://www.oddstrader.com/wnba/"
}

$Session = New-Object Microsoft.PowerShell.Commands.WebRequestSession

function Invoke-OddsTraderQuery {
    param(
        [Parameter(Mandatory = $true)][string]$Query,
        [int]$Attempts = 4
    )

    $encoded = [System.Uri]::EscapeDataString($Query)
    $url = "$ApiEndpoint?query=$encoded"
    $last = $null

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        try {
            $response = Invoke-RestMethod `
                -Uri $url `
                -Method Get `
                -Headers $Headers `
                -WebSession $Session `
                -TimeoutSec 60

            if ($response.errors) {
                $messages = @($response.errors | ForEach-Object { $_.message }) -join "; "
                throw "GraphQL error: $messages"
            }

            return $response
        }
        catch {
            $last = $_
            if ($attempt -lt $Attempts) {
                Start-Sleep -Seconds ([math]::Min(8, [math]::Pow(2, $attempt - 1)))
            }
        }
    }

    throw $last
}

function Initialize-OddsTraderSession {
    # A normal page hit establishes the same cookies/session a browser gets.
    try {
        Invoke-WebRequest `
            -Uri "https://www.oddstrader.com/wnba/?g=game&m=merged" `
            -Headers $Headers `
            -WebSession $Session `
            -TimeoutSec 60 `
            -UseBasicParsing | Out-Null
    }
    catch {
        Write-Warning "Initial WNBA page bootstrap failed; continuing with the structured endpoint: $($_.Exception.Message)"
    }
}

function Get-WnbaLeague {
    $query = @'
{
  leagues(enabled: true, limit: 500) {
    lid
    nam
    rid
    spid
    sn
    lurl
    settings {
      alias
      shortnamealias
    }
  }
}
'@

    $response = Invoke-OddsTraderQuery -Query $query
    $leagues = @($response.data.leagues)

    $matches = @(
        $leagues | Where-Object {
            $bits = @(
                $_.nam,
                $_.sn,
                $_.lurl,
                $_.settings.alias,
                $_.settings.shortnamealias
            ) -join " "

            $bits -match "(?i)(^|\W)WNBA(\W|$)|Women's National Basketball|Womens National Basketball"
        }
    )

    if ($matches.Count -eq 0) {
        throw "OddsTrader league discovery returned no WNBA league."
    }

    $exact = @(
        $matches | Where-Object {
            "$($_.sn)" -match "(?i)^WNBA$" -or
            "$($_.settings.alias)" -match "(?i)^WNBA$" -or
            "$($_.lurl)" -match "(?i)wnba"
        }
    )

    $league = if ($exact.Count -gt 0) { $exact[0] } else { $matches[0] }

    if (-not $league.lid -or -not $league.spid) {
        throw "WNBA league was found but lid/spid were missing."
    }

    return $league
}

function Get-EasternTimeZone {
    try {
        return [System.TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
    }
    catch {
        try {
            return [System.TimeZoneInfo]::FindSystemTimeZoneById("America/New_York")
        }
        catch {
            return [System.TimeZoneInfo]::Utc
        }
    }
}

function New-BookRow {
    param(
        [int]$Season,
        [string]$RequestedDate,
        $Event,
        [int64]$HomeId,
        [int64]$AwayId,
        [string]$HomeTeam,
        [string]$AwayTeam,
        [string]$HomeAbbr,
        [string]$AwayAbbr,
        [datetime]$LocalGameTime,
        [int]$Paid
    )

    $gameDate = $LocalGameTime.ToString("yyyy_MM_dd")
    $homeIdText = Normalize-TeamForId $HomeTeam
    $awayIdText = Normalize-TeamForId $AwayTeam

    return [ordered]@{
        sport                        = "basketball"
        league                       = "WNBA"
        season                       = $Season
        requested_date               = $RequestedDate
        game_date                    = $gameDate
        game_time                    = $LocalGameTime.ToString("hh:mm tt")
        oddstrader_event_id          = "$($Event.eid)"
        game_id                      = "WNBA_${gameDate}_${homeIdText}_${awayIdText}"
        home_team                    = $HomeTeam
        away_team                    = $AwayTeam
        home_abbr                    = $HomeAbbr
        away_abbr                    = $AwayAbbr
        home_participant_id          = $HomeId
        away_participant_id          = $AwayId
        sportsbook_id                = $Paid
        sportsbook                   = (Get-ProviderName $Paid)

        open_home_spread             = $null
        open_away_spread             = $null
        open_home_spread_american    = $null
        open_away_spread_american    = $null
        open_total                   = $null
        open_over_american           = $null
        open_under_american          = $null
        open_home_moneyline_american = $null
        open_away_moneyline_american = $null

        close_home_spread             = $null
        close_away_spread             = $null
        close_home_spread_american    = $null
        close_away_spread_american    = $null
        close_total                   = $null
        close_over_american           = $null
        close_under_american          = $null
        close_home_moneyline_american = $null
        close_away_moneyline_american = $null

        source_page = "https://www.oddstrader.com/wnba/?date=$($LocalGameTime.ToString('yyyyMMdd'))&g=game&m=merged"
        source_api  = $ApiEndpoint
    }
}

function Apply-Line {
    param(
        [hashtable]$Row,
        $Line,
        [ValidateSet("open", "close")][string]$Phase,
        [int64]$HomeId,
        [int64]$AwayId
    )

    if ($null -eq $Line.mtid) { return }

    $mtid = [int]$Line.mtid
    $partid = if ($null -ne $Line.partid) { [int64]$Line.partid } else { [int64]0 }
    $adj = $Line.adj
    $price = $Line.ap

    switch ($mtid) {
        401 {
            if ($partid -eq $HomeId) {
                $Row["${Phase}_home_spread"] = $adj
                $Row["${Phase}_home_spread_american"] = $price
            }
            elseif ($partid -eq $AwayId) {
                $Row["${Phase}_away_spread"] = $adj
                $Row["${Phase}_away_spread_american"] = $price
            }
        }

        402 {
            # OddsTrader/SBR participant IDs used for totals:
            # 15143 = Over, 15144 = Under.
            if ($partid -eq 15143) {
                $Row["${Phase}_total"] = $adj
                $Row["${Phase}_over_american"] = $price
            }
            elseif ($partid -eq 15144) {
                if ($null -eq $Row["${Phase}_total"]) {
                    $Row["${Phase}_total"] = $adj
                }
                $Row["${Phase}_under_american"] = $price
            }
        }

        83 {
            if ($partid -eq $HomeId) {
                $Row["${Phase}_home_moneyline_american"] = $price
            }
            elseif ($partid -eq $AwayId) {
                $Row["${Phase}_away_moneyline_american"] = $price
            }
        }
    }
}

function Get-DateEvents {
    param(
        [datetime]$Date,
        [int]$LeagueId,
        [int]$SportId
    )

    $utcMidnight = [DateTimeOffset]::new(
        $Date.Year, $Date.Month, $Date.Day, 0, 0, 0, [TimeSpan]::Zero
    )
    $timestampMs = $utcMidnight.ToUnixTimeMilliseconds()
    $providerList = ($ProviderIds -join ",")

    $query = @"
{
  eventsByDateByLeagueGroup(
    leagueGroups: [{ mtid: [401,402,83], lid: $LeagueId, spid: $SportId }],
    showEmptyEvents: true,
    marketTypeLayout: "PARTICIPANTS",
    ic: false,
    startDate: $timestampMs,
    timezoneOffset: -4,
    nof: true,
    hl: true,
    sort: { by: ["lid","dt","des"], order: ASC }
  ) {
    events {
      eid
      lid
      spid
      dt
      es
      participants {
        partid
        ih
        source {
          ... on Team {
            tmid
            nam
            nn
            sn
            abbr
            cit
          }
        }
      }
      currentLines(paid: [$providerList])
      openingLines
    }
  }
}
"@

    $response = Invoke-OddsTraderQuery -Query $query
    return @($response.data.eventsByDateByLeagueGroup.events)
}

function Has-AnyMarketValue {
    param([hashtable]$Row)

    $fields = @(
        "open_home_spread",
        "open_away_spread",
        "open_total",
        "open_home_moneyline_american",
        "open_away_moneyline_american",
        "close_home_spread",
        "close_away_spread",
        "close_total",
        "close_home_moneyline_american",
        "close_away_moneyline_american"
    )

    foreach ($field in $fields) {
        if ($null -ne $Row[$field] -and "$($Row[$field])".Trim() -ne "") {
            return $true
        }
    }

    return $false
}

function Complete-CurrentThreeMarkets {
    param($Row)

    return (
        $null -ne $Row.close_home_spread -and
        $null -ne $Row.close_away_spread -and
        $null -ne $Row.close_total -and
        $null -ne $Row.close_home_moneyline_american -and
        $null -ne $Row.close_away_moneyline_american
    )
}

$RepoRoot = Resolve-RepoRoot $RepoRoot
$OutDir = Join-Path $RepoRoot "docs\win\basketball\00_intake\sportsbook_history\oddstrader\wnba"
$RawDir = Join-Path $OutDir "raw"
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
New-Item -ItemType Directory -Path $RawDir -Force | Out-Null

$AllBooksCsv = Join-Path $OutDir "WNBA_2023_2024_ODDSTRADER_ALL_BOOKS.csv"
$DraftKingsCsv = Join-Path $OutDir "WNBA_2023_2024_ODDSTRADER_DRAFTKINGS.csv"
$CoverageTxt = Join-Path $OutDir "WNBA_2023_2024_ODDSTRADER_COVERAGE.txt"

Write-Host "=== WNBA 2023-2024 ODDSTRADER HISTORICAL ODDS BACKFILL ==="
Write-Host "Repo: $RepoRoot"
Write-Host "Output: $OutDir"
Write-Host ""

Initialize-OddsTraderSession

$league = Get-WnbaLeague
$LeagueId = [int]$league.lid
$SportId = [int]$league.spid

Write-Host "WNBA discovered: lid=$LeagueId spid=$SportId name=$($league.nam)"
Write-Host "Markets: spread=401 total=402 moneyline=83"
Write-Host "DraftKings provider id: 91"
Write-Host ""

$Ranges = @(
    [pscustomobject]@{
        Season = 2023
        Start  = [datetime]"2023-05-19"
        End    = [datetime]"2023-10-18"
    },
    [pscustomobject]@{
        Season = 2024
        Start  = [datetime]"2024-05-14"
        End    = [datetime]"2024-10-20"
    }
)

$Eastern = Get-EasternTimeZone
$allRows = New-Object System.Collections.Generic.List[object]
$seenEvents = @{}
$failures = New-Object System.Collections.Generic.List[string]

foreach ($range in $Ranges) {
    $season = [int]$range.Season
    $date = [datetime]$range.Start
    $seasonEventCount = 0

    Write-Host "[$season] scanning $($range.Start.ToString('yyyy-MM-dd')) through $($range.End.ToString('yyyy-MM-dd'))"

    while ($date -le $range.End) {
        $dateText = $date.ToString("yyyy-MM-dd")

        try {
            $events = @(Get-DateEvents -Date $date -LeagueId $LeagueId -SportId $SportId)

            if ($events.Count -gt 0) {
                $rawPath = Join-Path $RawDir "$($date.ToString('yyyyMMdd')).json"
                $events | ConvertTo-Json -Depth 30 | Set-Content -Path $rawPath -Encoding UTF8
            }

            foreach ($event in $events) {
                if (-not $event.eid) { continue }

                $eventKey = "$season|$($event.eid)"
                if ($seenEvents.ContainsKey($eventKey)) { continue }

                $participants = @($event.participants)
                $home = $participants | Where-Object { $_.ih -eq $true -and $null -ne $_.source.tmid } | Select-Object -First 1
                $away = $participants | Where-Object { $_.ih -ne $true -and $null -ne $_.source.tmid } | Select-Object -First 1

                if ($null -eq $home -or $null -eq $away) {
                    continue
                }

                $seenEvents[$eventKey] = $true
                $seasonEventCount++

                $homeId = [int64]$home.source.tmid
                $awayId = [int64]$away.source.tmid
                $homeTeam = "$($home.source.nam)"
                $awayTeam = "$($away.source.nam)"
                $homeAbbr = "$($home.source.abbr)"
                $awayAbbr = "$($away.source.abbr)"

                $utc = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$event.dt).UtcDateTime
                $localGameTime = [System.TimeZoneInfo]::ConvertTimeFromUtc($utc, $Eastern)

                $bookRows = @{}

                foreach ($phaseInfo in @(
                    [pscustomobject]@{ Phase = "open";  Lines = @($event.openingLines) },
                    [pscustomobject]@{ Phase = "close"; Lines = @($event.currentLines) }
                )) {
                    foreach ($line in $phaseInfo.Lines) {
                        if ($null -eq $line -or $null -eq $line.paid) { continue }

                        $paid = [int]$line.paid
                        if (-not $bookRows.ContainsKey($paid)) {
                            $bookRows[$paid] = New-BookRow `
                                -Season $season `
                                -RequestedDate $dateText `
                                -Event $event `
                                -HomeId $homeId `
                                -AwayId $awayId `
                                -HomeTeam $homeTeam `
                                -AwayTeam $awayTeam `
                                -HomeAbbr $homeAbbr `
                                -AwayAbbr $awayAbbr `
                                -LocalGameTime $localGameTime `
                                -Paid $paid
                        }

                        Apply-Line `
                            -Row $bookRows[$paid] `
                            -Line $line `
                            -Phase $phaseInfo.Phase `
                            -HomeId $homeId `
                            -AwayId $awayId
                    }
                }

                foreach ($paid in $bookRows.Keys) {
                    $row = $bookRows[$paid]
                    if (Has-AnyMarketValue $row) {
                        $allRows.Add([pscustomobject]$row)
                    }
                }
            }

            if ($events.Count -gt 0) {
                Write-Host "  $dateText : events=$($events.Count) cumulative_unique=$seasonEventCount"
            }
        }
        catch {
            $msg = "$dateText | $($_.Exception.Message)"
            $failures.Add($msg)
            Write-Warning $msg
        }

        Start-Sleep -Milliseconds $DelayMilliseconds
        $date = $date.AddDays(1)
    }

    Write-Host "[$season] unique WNBA events found: $seasonEventCount"
    Write-Host ""
}

# Remove any accidental duplicate event/book rows and sort.
$allRowsFinal = @(
    $allRows |
        Sort-Object season, game_date, oddstrader_event_id, sportsbook_id -Unique
)

if ($allRowsFinal.Count -eq 0) {
    throw "No OddsTrader odds rows were extracted. Raw failures, if any, are in the console output."
}

$allRowsFinal |
    Export-Csv -Path $AllBooksCsv -NoTypeInformation -Encoding UTF8

# Produce a DraftKings file compatible with the market columns used by 2025_WNBA.csv.
$dkRows = @(
    $allRowsFinal |
        Where-Object { [int]$_.sportsbook_id -eq 91 } |
        ForEach-Object {
            [pscustomobject][ordered]@{
                sport                         = "basketball"
                league                        = "WNBA"
                season                        = $_.season
                game_date                     = $_.game_date
                game_id                       = $_.game_id
                odds_last_update               = ""
                game_time                     = $_.game_time
                home_team                     = $_.home_team
                away_team                     = $_.away_team

                # Historical final/current line from OddsTrader.
                home_spread                   = $_.close_home_spread
                away_spread                   = $_.close_away_spread
                total                         = $_.close_total
                home_dk_moneyline_american    = $_.close_home_moneyline_american
                away_dk_moneyline_american    = $_.close_away_moneyline_american
                home_dk_spread_american       = $_.close_home_spread_american
                away_dk_spread_american       = $_.close_away_spread_american
                dk_total_over_american        = $_.close_over_american
                dk_total_under_american       = $_.close_under_american

                home_dk_moneyline_decimal     = Convert-AmericanToDecimal $_.close_home_moneyline_american
                away_dk_moneyline_decimal     = Convert-AmericanToDecimal $_.close_away_moneyline_american
                home_dk_spread_decimal        = Convert-AmericanToDecimal $_.close_home_spread_american
                away_dk_spread_decimal        = Convert-AmericanToDecimal $_.close_away_spread_american
                dk_total_over_decimal         = Convert-AmericanToDecimal $_.close_over_american
                dk_total_under_decimal        = Convert-AmericanToDecimal $_.close_under_american

                # Preserve the opener too; Step 18 can ignore these if not needed.
                open_home_spread              = $_.open_home_spread
                open_away_spread              = $_.open_away_spread
                open_total                    = $_.open_total
                open_home_dk_moneyline_american = $_.open_home_moneyline_american
                open_away_dk_moneyline_american = $_.open_away_moneyline_american
                open_home_dk_spread_american  = $_.open_home_spread_american
                open_away_dk_spread_american  = $_.open_away_spread_american
                open_dk_total_over_american   = $_.open_over_american
                open_dk_total_under_american  = $_.open_under_american

                oddstrader_event_id           = $_.oddstrader_event_id
                odds_source                   = "OddsTrader"
                odds_source_page              = $_.source_page
                odds_source_api               = $_.source_api
            }
        }
)

$dkRows |
    Export-Csv -Path $DraftKingsCsv -NoTypeInformation -Encoding UTF8

# Coverage audit.
$report = New-Object System.Collections.Generic.List[string]
$report.Add("WNBA 2023-2024 ODDSTRADER COVERAGE")
$report.Add("generated_at_utc=$([DateTime]::UtcNow.ToString('o'))")
$report.Add("wnba_lid=$LeagueId")
$report.Add("basketball_spid=$SportId")
$report.Add("market_ids=spread:401,total:402,moneyline:83")
$report.Add("draftkings_provider_id=91")
$report.Add("")

foreach ($season in @(2023, 2024)) {
    $seasonAll = @($allRowsFinal | Where-Object { [int]$_.season -eq $season })
    $seasonDk = @($dkRows | Where-Object { [int]$_.season -eq $season })

    $allEventIds = @($seasonAll | Select-Object -ExpandProperty oddstrader_event_id -Unique)
    $dkEventIds = @($seasonDk | Select-Object -ExpandProperty oddstrader_event_id -Unique)
    $dkComplete = @($seasonDk | Where-Object { Complete-CurrentThreeMarkets $_ })
    $anyCompleteEventIds = @(
        $seasonAll |
            Where-Object { Complete-CurrentThreeMarkets $_ } |
            Select-Object -ExpandProperty oddstrader_event_id -Unique
    )

    $report.Add("$season")
    $report.Add("  unique_events_with_any_book_odds=$($allEventIds.Count)")
    $report.Add("  unique_events_with_draftkings=$($dkEventIds.Count)")
    $report.Add("  draftkings_events_complete_ml_spread_total=$($dkComplete.Count)")
    $report.Add("  any_book_events_complete_ml_spread_total=$($anyCompleteEventIds.Count)")
    $report.Add("")
}

$report.Add("all_books_rows=$($allRowsFinal.Count)")
$report.Add("draftkings_rows=$($dkRows.Count)")
$report.Add("request_failures=$($failures.Count)")

if ($failures.Count -gt 0) {
    $report.Add("")
    $report.Add("FAILURES")
    foreach ($failure in $failures) {
        $report.Add("  $failure")
    }
}

$report | Set-Content -Path $CoverageTxt -Encoding UTF8

Write-Host ""
Write-Host "=== COMPLETE ==="
Write-Host "All books:  $AllBooksCsv"
Write-Host "DraftKings: $DraftKingsCsv"
Write-Host "Coverage:   $CoverageTxt"
Write-Host ""
Write-Host "Rows: all_books=$($allRowsFinal.Count) draftkings=$($dkRows.Count) failures=$($failures.Count)"
Write-Host "markets.yaml modified: NO"
Write-Host "Step 18 files modified: NO"
