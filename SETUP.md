# Cryptobot — GitHub-hosted versioon (24/7, ilma et su arvuti peaks sees olema)

Kõik käib github.com veebilehe kaudu — käsurida pole vaja.

## 1. Telegrami bot loomine (5 min)

1. Ava Telegramis vestlus kasutajaga **@BotFather**.
2. Saada `/newbot`, anna botile nimi ja kasutajanimi (peab lõppema sõnaga "bot", nt `RandoCryptoBot`).
3. BotFather saadab sulle **tokeni** (kujul `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`) — kopeeri see kuskile.
4. Ava oma uue botiga vestlus (link on BotFather sõnumis) ja saada talle mistahes sõnum (nt "hei").
5. Ava brauseris (asenda TOKEN oma tokeniga):
   `https://api.telegram.org/botTOKEN/getUpdates`
   Tulemuse seest leiad `"chat":{"id": 123456789, ...}` — see number ongi su **chat_id**.

## 2. GitHubi repo loomine — GitHub Desktopiga (soovitatud, kui laed GitHubi arvutisse)

Kui pigem installid GitHubi enda arvutisse, siis kõige lihtsam viis on **GitHub Desktop** (ametlik GitHubi rakendus, käsurida pole vaja):

1. Laadi alla ja installi: [desktop.github.com](https://desktop.github.com)
2. Ava rakendus ja logi sisse oma GitHubi kontoga.
3. Vali menüüst **File → Add Local Repository**.
4. Vali kaust: see `hosted-bot` kaust, mis on Finderis siin: `Desktop/skrrrando/rahategemine/cryptobot/hosted-bot` (kõik failid - `engine.py`, `watchlist.json`, `fetch_and_run.py`, `.github/workflows/hourly.yml`, `data/` - on juba seal olemas, valmis kujul).
5. GitHub Desktop ütleb, et see kaust pole veel git-repo, ja pakub "create a repository" nuppu — vajuta seda.
6. Vajuta "Create Repository" (nimi ja asukoht on juba täidetud).
7. Üleval vajuta suurt sinist nuppu **"Publish repository"**. Märgi "Keep this code private" ja vajuta "Publish Repository".
8. Valmis — kõik failid (koos kaustastruktuuriga) on nüüd sinu GitHubi kontol.

### Alternatiiv: ilma GitHub Desktopita, otse veebis

1. Mine [github.com/new](https://github.com/new), loo uus repo (nt nimega `cryptobot`), vali **Private**, vajuta "Create repository".
2. Repo lehel vajuta "Add file" → "Upload files".
3. Lohista sellest kaustast (`hosted-bot/`) SEES olevad failid ja kaustad üles: `engine.py`, `watchlist.json`, `fetch_and_run.py`, `.github` (kogu kaust koos `workflows/hourly.yml`-ga), `data` (kogu kaust). GitHub säilitab kaustastruktuuri, kui lohistad terve kausta korraga.
4. Vajuta "Commit changes".

## 3. Telegrami andmete lisamine (secrets)

1. Repo lehel: **Settings** → vasakul **Secrets and variables** → **Actions**.
2. Vajuta "New repository secret":
   - Name: `TELEGRAM_BOT_TOKEN`, Value: (samm 1 token)
   - Vajuta "Add secret"
3. Korda: Name: `TELEGRAM_CHAT_ID`, Value: (samm 1 chat_id)

## 4. Käivitamine

1. Mine repo **Actions** vahekaardile. Kui GitHub küsib kinnitust ("I understand my workflows, go ahead and enable them"), vajuta seda.
2. Vali vasakult "Cryptobot hourly scan", vajuta paremal "Run workflow" → "Run workflow" — see käivitab kohe esimese testkäigu (ei pea ootama tundi).
3. Mõne minuti pärast peaksid saama Telegramis esimese sõnumi. Kui ei tule, vajuta käigule (töö nimekirjas) ja vaata logi — enamasti on põhjus vale token/chat_id.

Pärast seda jookseb see täiesti iseseisvalt iga tund, olenemata sellest, kas su arvuti on sees või mitte.

## Mis on selles versioonis teisiti kui Coworki-siseses versioonis

- **Hype-kontroll veebist puudub siin** — GitHub'i serveril pole minu veebiotsingu tööriista, seega skoor põhineb ainult turuandmetel (momentum + trend + likviidsus/scam-heuristika), mitte enam ka Twitteri/uudiste kinnitusel. Kui tahad seda hiljem tagasi, saame lisada tasulise uudiste-API (räägime siis).
- Teavitus tuleb nüüd **Telegrami**, mitte Cowork vestlusesse.
- `data/` kaust ja `dashboard.html` uuenevad otse selles GitHub repos igal käigul — saad need alati sealt vaadata või alla laadida.

## Turvalisus

See bot AINULT jälgib ja soovitab — see ei osta ega müü kunagi midagi automaatselt.
