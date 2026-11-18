# Senti-Q for X (browser extension)

Analyse the post you're looking at on X with one click, using your own logged-in X
in your own browser. Works in Chrome, Edge, Brave and other Chromium browsers.

## Install (about 1 minute)

1. Start Senti-Q (double-click `run.bat` in the project folder).
2. Open `chrome://extensions` (Edge: `edge://extensions`).
3. Turn on **Developer mode** (top-right in Chrome, left sidebar in Edge).
4. Click **Load unpacked** and select this `browser-extension` folder.
5. Pin **Senti-Q for X** to your toolbar (puzzle-piece icon → pin).

## Use

* **On a post's page:** click the Senti-Q icon to analyse that post.
* **Scrolling your timeline:** click the icon to analyse the post in the middle of
  your screen.
* **Anywhere on X:** right-click a post's timestamp → **Analyse with Senti-Q**.

Senti-Q opens in a new tab with the post already analysed.

## Privacy

The extension never touches your X password, cookies or account data. When you click,
it reads the link of that one post and opens `http://localhost:8501/?post=<link>`.
Senti-Q then fetches the public post through X's embed endpoint. To point it at a
different Senti-Q address, right-click the icon → **Options**.
