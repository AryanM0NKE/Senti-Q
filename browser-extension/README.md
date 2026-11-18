# Senti-Q for X (browser extension)

See how people are reacting to a post on X. The extension reads the replies you can see
on a post's page, in your own signed-in browser, and opens Senti-Q with them. Works in
Chrome, Edge, Brave and other Chromium browsers.

## Install (about 1 minute)

1. Start Senti-Q (double-click `run.bat` in the project folder).
2. Open `chrome://extensions` (Edge: `edge://extensions`).
3. Turn on **Developer mode** (top-right in Chrome, left sidebar in Edge).
4. Click **Load unpacked** and select this `browser-extension` folder.
5. Pin **Senti-Q for X** to your toolbar (puzzle-piece icon → pin).
6. Reload any X tabs that were already open.

Updating: after pulling a new version, click the reload icon on the extension's card in
`chrome://extensions`, then reload your X tabs.

## Use

* **From Senti-Q:** paste a post link, press **Analyse**, then **Read the replies on X**.
  The post opens, the replies are read (a small black pill shows progress), the tab closes
  and you're back in Senti-Q with the results.
* **On a post's page:** click the Senti-Q icon.
* **Scrolling a timeline:** click the icon to read the post in the middle of your screen.
* **Anywhere on X:** right-click a post's timestamp → **Read the reactions with Senti-Q**.

**Options** (right-click the icon → Options): the Senti-Q address (default
`http://localhost:8501`) and how many replies to read per post (50–300, default 100).

## What it reads, and what it doesn't

* **When:** only when you ask (a click, or a link from Senti-Q ending in `#senti-q`), and
  only on that post's page.
* **What:** the post and, for each reply, its text, @handle, like count and language.
  Replies hidden behind *Show more replies* / *Show probable spam* are revealed and
  included.
* **Left out:** the thread above the post, the author's own replies, ads, image-only
  replies and X's unrelated *Discover more* posts.
* **Never:** your password, cookies, messages or account settings. It makes no network
  requests of its own. The replies travel to Senti-Q inside the link it opens, and
  nothing is stored.

The permissions it asks for are for x.com / twitter.com (to read the replies) and
localhost (to find your Senti-Q tab).
