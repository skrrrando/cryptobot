# Cryptobot — kokkuvõte (seisuga 15.07.2026)

Isiklik krüpto-signaali bot Randole. **Ei osta ega müü kunagi päris raha eest** — ainult soovitab ja jälgib.

## Kus see jookseb

- Kood: GitHub repo `skrrrando/cryptobot` (**Public**, kuna GitHub Pages tasuta plaan nõuab avalikku repot; git ajalugu on puhastatud päris e-mailist).
- Käivitub: GitHub Actions, iga tund (`0 * * * *` UTC), täiesti iseseisvalt — Rando arvuti ei pea sees olema.
- Dashboard: **https://skrrrando.github.io/cryptobot/** (uueneb iga tunnise käigu järel automaatselt).
- Teavitus: Telegrami bot ("ESTUP coin"), token/chat_id on GitHubi encrypted Secrets all (mitte koodis).

## Kuidas skoor tekib

1. Tõmbab reaalajas hinnad/mahud otse Crypto.com avalikust API'st (`fetch_and_run.py`), 45 tokeni kohta (`watchlist.json` — 28 "majors" + 17 "meme_trend").
2. `engine.py screen`: momentum (24h % muutus) + trendi kinnitus (kas hind/maht on tõusnud järjest mitu tundi, mitte ühekordne hüpe) → raw skoor. Läve ületajad lähevad edasi. Lisaks väike "eksperimentaalne" valik (epsilon-greedy, ~15%) allpool läve, et mudel õpiks ka piiripealsetest juhtudest.
3. `engine.py finalize`: lisab riskimärgistuse (roheline/kollane/punane — likviidsus + volatiilsus + kategooria), kombineerib käsitsi-reeglitega skoori isikliku **õppiva mudeliga** (logistiline regressioon, käsitsi kirjutatud Python, treenitud iga kord kui üks soovitus 24h pärast tulemuse saab). Mudel mõjutab skoori alles pärast 15 treeningtulemust.
4. **Praegu EI OLE reaalset hype/Twitteri-kontrolli hostitud versioonis** — GitHub Actions ei saa kasutada veebiotsingu tööriista, seega skoor põhineb ainult turuandmetel + õppival mudelil. (Cowork-sisene versioon algselt tegi ka WebSearchi, aga see disaini-osa jäi hostitud versiooni juures kõrvale.)

## Isikliku õppimise osad

- **Tagasivaade**: iga soovitus saab 24h ja 7p pärast tulemuse (kas hind läks üles/alla), salvestatakse.
- **Õppiv mudel**: kaalud (momentum, trend, likviidsus, meemi-kategooria, volatiilsus, hype) liiguvad selle poole, mis päriselt tabamist ennustab (koos L2 regularisatsiooniga müra vastu).
- **Adaptiivsed lävendid**: kui roheline/kollane/punane risk tabab liiga harva/tihti, tõuseb/langeb lävi — aga ainult siis, kui on PÄRISELT uut infot (parandatud viga, mis muidu iga tund sama vana tõendi peale uuesti reageeris).
- **Õpipäevik**: tavakeelne logi dashboardil, mida mudel viimati enda kohta õppis.

## Virtuaalne portfell (mängu raha)

- Algsaldo $1000 (fiktiivne).
- Iga alerti peale "ostab" 5% praegusest saldost, max 5 positsiooni korraga.
- "Müüb" automaatselt 24h pärast, päris hinnaliikumise järgi.
- Dashboard näitab saldograafikut, avatud/suletud kauplusi, ja lihtsat plussis/miinuses kokkuvõtet.
- Eesmärk: kui saldo aja jooksul püsivalt kasvab, on see märk et tasuks kaaluda päris raha (Rando enda otsus, bot ei tee seda kunagi ise).

## Teadaolevad valikud/kompromissid

- LunarCrush (päris hype-andmed) nõuaks $90/kuu — jäeti kõrvale, kasutatakse ainult turuandmeid.
- Otsene Telegram/API juurdepääs polnud Coworki liivakastis võimalik → lahenduseks GitHub Actions + otse-API kutsed.
- Vana Cowork-sisene ajastatud ülesanne ("cryptobot-hourly-scan") on **disabled**, asendatud GitHub Actionsiga.

## Kui tahad edasi arendada

- Watchlist (`watchlist.json`) on vabalt muudetav GitHubis.
- Kui kunagi tahad päris hype-andmeid tagasi, tuleks kaaluda tasulist uudiste/social-API-t (LunarCrush vms).
- Portfelli reegleid (5% suurus, 24h hoid, max 5) saab `engine.py` konstantidest muuta.
