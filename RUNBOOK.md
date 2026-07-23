# Cryptobot — tunniplaani juhend

See on käsiraamat, mida ajastatud ülesanne (scheduled task) iga tund täpselt järgib. Kõik failid asuvad selles kaustas (`cryptobot/`), mitte ajutises töökaustas — nii säilib ajalugu tundide vahel.

## Failid
- `engine.py` — skoorimismootor (screen + finalize) + portfellihaldus.
- `broker.py` — täitmiskiht. ÜKS liides, kaks implementatsiooni: `PaperBroker` (simulatsioon koos teenustasude + slippage'iga) ja `LiveBroker` (päris orderid Crypto.com Exchange'il). Valik käib `TRADING_MODE` env muutujaga (`paper` on vaikimisi).
- `watchlist.json` — jälgitavad tokenid (major + meme/trend kategooriad). Vabalt muudetav.
- `data/state.json` — kogu ajalugu, ootel/lõpetatud soovitused, adaptiivsed lävendid, portfell, kill-switch. EI kustutata kunagi, ainult uuendatakse.
- `data/instrument_map.json` — watchlisti sümbol → börsi päris instrumendinimi (uuendatakse igal käivitusel; LiveBroker vajab orderite jaoks).
- `dashboard.html` — visuaalne ülevaade, genereeritakse iga käivituse lõpus uuesti.

## Iga tunni sammud (mida agent teeb)

1. **Tooma turuandmed** — kutsu Crypto.com `get_tickers` iga `watchlist.json`-is oleva instrumendi kohta (üks kutse per sümbol, ~45 kutset). Pane tulemused kokku JSON-listiks kujul `[{"instrument_name":..., "last":..., "change":..., "volume_value":...}, ...]` ja kirjuta faili (nt `data/tickers_latest.json`).

2. **Stage 1 — screen** — käivita:
   `python3 engine.py screen --tickers data/tickers_latest.json --out data/candidates.json`
   See uuendab `data/state.json` ajalugu ja kirjutab välja kuni 10 kandidaati, kes ületasid praeguse skanni läve.

3. **Hype-kontroll (ainult kandidaatidele, mitte kõigile 45-le)** — kui `data/candidates.json` pole tühi, tee iga kandidaadi kohta üks `WebSearch` (nt "`<token nimi>` crypto news today"). Otsusta lühidalt: `found` (kas midagi värsket leidus), `sentiment` (`positive`/`neutral`/`negative`/`warning` — `warning` kui leiad rug pull/scam/hack/hoiatuse viiteid), `summary` (1-2 lauset). Kirjuta tulemus faili `data/hype_notes.json` kujul `{"<instrument>": {"found":true,"sentiment":"positive","summary":"..."}}`.

4. **Tagasivaate hinnad** — vaata `data/state.json` väljast `pending_followups`. Kui seal on kirjeid, mille `ts` on >=24h või >=7 päeva vana, too nende instrumentide jaoks värske hind (`get_tickers`) ja kirjuta `data/current_prices.json` kujul `{"<instrument>": price}`. Kui pending_followups on tühi, kirjuta `{}`.

5. **Stage 2 — finalize** — käivita:
   `python3 engine.py finalize --candidates data/candidates.json --hype-notes data/hype_notes.json --current-prices data/current_prices.json --out-summary data/summary_latest.txt`
   See arvutab lõpliku skoori + riski + põhjenduse, rakendab cooldown/dedup (ei tüüta korduvate teadetega), logib uued soovitused, lahendab tähtaja ületanud tagasivaated, kohandab vajadusel lävendeid, ja kirjutab uue `dashboard.html`.

6. **Teade kasutajale** — loe `data/summary_latest.txt` ja postita selle sisu Cowork vestlusesse (see ongi "teavitus telefonis", kui push on sees). Kui soovid, lisa `present_files` kutse dashboard.html jaoks (mitte iga tund, piisab kord mõne tunni jooksul või kui kasutaja küsib).

## Kauplemisrežiimid

Sama kood, kaks režiimi — see on kogu süsteemi selgroog. Kuu aega paper-režiimis jooksmine valideerib TÄPSELT sama süsteemi, mis hiljem päris rahaga kaupleb.

| Env muutuja | Vaikimisi | Mida teeb |
|---|---|---|
| `TRADING_MODE` | `paper` | `paper` = simulatsioon, `live` = päris orderid Crypto.com'il |
| `TRADING_FEE_PCT` | `0.5` | Taker fee % ühe poole kohta. Sea oma tegeliku taseme järgi. |
| `TRADING_SLIPPAGE_PCT` | `0.15` | Simuleeritud slippage % (ainult paper-fillidel) |
| `STOP_LOSS_PCT` | `8` | Stop-loss % sisenemishinnast allpool |
| `MAX_DAILY_LOSS_PCT` | `5` | Kill-switch: max lubatud 24h kaotus % |
| `MAX_CONSEC_FAILURES` | `3` | Kill-switch: max järjestikuseid ebaõnnestunud ordereid |
| `KILLSWITCH_RESET` | – | `1` = lähtesta aktiivne kill-switch selle käivitusega |
| `CRYPTO_API_KEY` / `CRYPTO_API_SECRET` | – | Ainult live-režiimis. AINULT GitHub Secrets kaudu! |

### Kaitsemehhanismid (mõlemas režiimis)
- **Stop-loss**: iga ostuga pannakse kaasa stop -8% sisenemisest. Live-režiimis on see PÄRIS order börsi poolel (kaitse ei sõltu sellest, kas tunnine cron õigel ajal jookseb); paper-režiimis simuleeritakse igal käivitusel.
- **Kill-switch**: kui portfell kaotab >5% 24h jooksul VÕI 3 orderit järjest ebaõnnestub, lõpetab bot uute positsioonide avamise (olemasolevaid haldab edasi) ja ütleb seda Telegramis. Lähtestamine on teadlik käsitsi samm: käivita üks kord `KILLSWITCH_RESET=1`-ga.
- **Neto-P&L**: kõik tulemused (ka paper) sisaldavad teenustasusid mõlemal pool tehingut, et kuu testiperioodi numbrid ei oleks ilustatud.

## Go-live checklist (ÄRA jäta ühtegi sammu vahele)

1. **Paper-faas: vähemalt 30 päeva** `TRADING_MODE=paper` (vaikimisi, midagi tegema ei pea). Edukriteeriumid, mis peavad KÕIK täidetud olema enne raha:
   - profit factor > 1.3
   - max drawdown < 15%
   - vähemalt 60 suletud kauplust
   - positiivne expectancy (oodatav väärtus/kauplus > 0) — NB! see on juba fee-järgne number
   - Kui kasvõi üks jääb puudu → raha ei lähe sisse. Pikenda paper-faasi või paranda strateegiat.
2. **API võti**: loo Crypto.com Exchange'is API võti, millel on AINULT kauplemisõigus — väljamaksed (withdrawal) KEELATUD. Lisa IP-piirang kui võimalik.
3. **Secrets**: lisa GitHub repo Settings → Secrets and variables → Actions alla `CRYPTO_API_KEY` ja `CRYPTO_API_SECRET`. MITTE KUNAGI faili ega repo'sse — repo on avalik (GitHub Pages).
4. **Workflow env** (`.github/workflows/hourly.yml`, run-stepi külge):
   ```yaml
   env:
     TRADING_MODE: live        # alusta väärtusega "paper", vaheta alles pärast checklisti!
     TRADING_FEE_PCT: "0.5"    # sea oma tegeliku fee-taseme järgi
     CRYPTO_API_KEY: ${{ secrets.CRYPTO_API_KEY }}
     CRYPTO_API_SECRET: ${{ secrets.CRYPTO_API_SECRET }}
     TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
     TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
   ```
5. **Smoke-test väikese summaga**: enne täismahtu pane kontole ainult väike summa (nt 50–100 USD) ja jälgi 2–3 päeva käsitsi: kas orderid täituvad, kas stop-lossid lähevad üles, kas dashboard/Telegram klapivad börsi ajalooga. `broker.py` LiveBroker on kirjutatud Crypto.com Exchange API v1 järgi, aga POLE veel päris kontoga läbi proovitud — see samm on kohustuslik.
6. **Täissumma**: ainult raha, mille kaotamine ei tee haiget. Kelly ülempiir (15% per positsioon) ja max 5 positsiooni jäävad kehtima.

## Ausad ootused
Tunnipõhine momentum-strateegia on pärast teenustasusid enamasti negatiivse ootusega — dashboard näitab nüüd fee-järgset tõde. Usalda profit factorit ja expectancy't, mitte lootust. 30 päeva on üks turufaas, mitte lõplik tõestus: bull-turul õpitud kaalud võivad bear-turul valed olla. Kill-switch ja stop-lossid piiravad kahju, aga ei garanteeri midagi.

## Turvalisus
- Paper-režiimis (vaikimisi) EI puuduta süsteem päris raha mitte kunagi.
- Live-režiimis kaupleb bot automaatselt — sellepärast on stop-lossid, kill-switch, positsioonipiirangud ja go-live checklist kohustuslikud, mitte soovituslikud.
- API võtmed elavad AINULT GitHub Secrets'is, withdrawal-õiguseta.
