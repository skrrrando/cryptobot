# Cryptobot — tunniplaani juhend

See on käsiraamat, mida ajastatud ülesanne (scheduled task) iga tund täpselt järgib. Kõik failid asuvad selles kaustas (`cryptobot/`), mitte ajutises töökaustas — nii säilib ajalugu tundide vahel.

## Failid
- `engine.py` — skoorimismootor (screen + finalize) + portfellihaldus.
- `broker.py` — täitmiskiht. ÜKS liides, kaks implementatsiooni: `PaperBroker` (simulatsioon koos teenustasude + slippage'iga) ja `LiveBroker` (päris orderid Crypto.com Exchange'il). Valik käib `TRADING_MODE` env muutujaga (`paper` on vaikimisi).
- `watchlist.json` — jälgitavad tokenid (major + meme/trend kategooriad). Vabalt muudetav.
- `data/state.json` — kogu ajalugu, ootel/lõpetatud soovitused, adaptiivsed lävendid, portfell, kill-switch. EI kustutata kunagi, ainult uuendatakse.
- `data/instrument_map.json` — watchlisti sümbol → börsi päris instrumendinimi (uuendatakse igal käivitusel; LiveBroker vajab orderite jaoks).
- `data/candles_latest.json` — 24h jagu 15m OHLCV küünlaid iga instrumendi kohta (transient, gitignore'itud). Volatiilsus ja trend arvutatakse nendelt (96 punkti), hetktõmmised on varuvariant.
- `data/book_notes.json` — orderiraamatu imbalance + spread Stage-1 kandidaatidele (transient). Lai spread (>1%) annab skooritrahvi; imbalance on mudeli tunnus.
- `data/market_regime.json` — Fear & Greed indeks (transient). Mudeli tunnus `market_fng` + hoiatus summary's äärmuste (≤25 / ≥75) puhul.
- `data/funding_rates.json` — iga instrumendi perpetual'i annualiseeritud funding määr (transient). Toidab funding-arb sahtlit.
- `data/onchain_latest.json` + `data/onchain_history.json` — BTC on-chain tehingumaht ja selle 7-päeva keskmise suhe (transient). Mudeli tunnus `onchain_activity`.
- `backtest.py` — standalone tööriist (EI jookse tunnise workflow osana): mängib sama screen/finalize mootorit läbi ajaloolise andmega (`python3 backtest.py --months 6`), jaotab tulemused bull/bear/chop režiimide kaupa. Ei puuduta `data/state.json` - kirjutab eraldi `backtest_data/` kausta.
- `dashboard.html` — visuaalne ülevaade, genereeritakse iga käivituse lõpus uuesti.

## Positsioonihaldus (V5)
- **Osaline kasumivõtt**: +8% juures müüakse pool positsiooni (kasum pangas), teine pool jookseb edasi.
- **Trailing stop**: kui tipp on ≥ +4% sisenemisest, järgneb stop 3% kaugusel tipust (ainult ülespoole). Live-režiimis cancel+replace päris stop-order.
- **Vahetusloogika**: täis raamatu korral vahetatakse nõrgim ≥1% miinuses positsioon välja, kui uus kandidaat on ≥12 punkti tugevam (max 1/tunnis).
- **Täis-treening**: iga 20 uue lahendatud tulemuse järel treenitakse mudel nullist kogu ajaloo peal uuesti (150 epohhi) - stabiilsemad kaalud kui ainult ükshaaval õppides.
- **Korrelatsiooni piirmäär**: max 3/5 avatud positsioonist tohib olla madala-alphaga (tugevalt BTC-korreleeritud) korraga - "5 diversifitseeritud positsiooni" ei tohi salaja olla üks suur BTC-panus.

## Uued sahtlid (V6) — eraldi kapital, eraldi ledger, momentum-portfelli ei puuduta

**💹 Funding-rate arbitraaž** (`state["funding_arb"]`, algsaldo $300): turuneutraalne — ostab spot + avab võrdse suurusega lühikese perpetual-positsiooni (delta-neutraalne), teenib ainult funding-makseid, ei sõltu turu suunast.
- Sisenemine: annualiseeritud funding ≥ 20%. See lävi EI ole meelevaldne — nelja jala (spot ost/müük + perp ava/sule) teenustasu+slippage kulu on ~1% kapitalist, ja kuna funding koguneb ainult ühe jala (lühikese) notionali pealt, kulub tasuvuseni ~38 päeva 20% APR juures. Madalam lävi ei jõuaks 45-päevase max hoiuaja jooksul kunagi tasuvusse.
- **Vajab derivatiivide/marginaalkauplemise õigust Crypto.com kontol live-režiimis** — see on eraldi eeldusõigus spot-kauplemisest, võib vajada täiendavat KYC-d. Kontrolli enne live-minekut.
- Väljumine: funding langeb alla 1% APR, või 45 päeva täis.

**🔲 Grid/mean-reversion** (`state["grid"]`, algsaldo $300): likviidsetel majoritel (BTC, ETH) — ostab kui hind on oma 24h vahemiku põhjas 35% JA trend on nõrk (R²<0.35, momentumi vastand), müüb +3% juures või vahemiku tipus, stop -6%. Monetiseerib tunde, mil momentum-strateegia lihtsalt ootab (enamik aega on crypto turg vahemikus, mitte trendis).

Mõlemad on gate'itud sama kill-switchiga kui momentum-portfell (konto-tasandi kaitse peab peatama KOGU uue riski, mitte ainult ühe strateegia oma).

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
