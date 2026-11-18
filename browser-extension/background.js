// Senti-Q for X — background service worker.
//
// * Toolbar icon on a post's page: reads that post's replies right there.
// * Toolbar icon anywhere else on X: pick mode. Click the post you want; it opens in the
//   same tab, its replies are read, and you're taken back to where you were.
// * Right-click a post's link -> "Read the reactions with Senti-Q" (same as picking it).
// * Senti-Q's "Read the replies on X" links end in #senti-q: the replies are read
//   automatically, that X tab closes, and you're back in the Senti-Q tab you came from.
//
// Results always go to one Senti-Q tab (reused, not a new tab every time). The replies
// travel inside the Senti-Q link itself (compressed), so nothing is stored or sent elsewhere.

const DEFAULTS = { appUrl: "http://localhost:8501", maxReplies: 100 };
const MAX_TOKEN = 24000;   // characters of packed replies in the Senti-Q link
const POST_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\/(?:i\/web|[^/?#]+)\/status\/(\d+)/;
const X_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\//;

async function appBase() {
  const { appUrl } = await chrome.storage.sync.get(DEFAULTS);
  return appUrl.replace(/\/+$/, "");
}

const safeDecode = (s) => { try { return decodeURIComponent(s); } catch (e) { return s; } };

// ---------- packing: JSON -> raw deflate -> base64url ----------
async function deflateBase64(obj) {
  const bytes = new TextEncoder().encode(JSON.stringify(obj));
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate-raw"));
  const buf = new Uint8Array(await new Response(stream).arrayBuffer());
  let bin = "";
  for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode(...buf.subarray(i, i + 0x8000));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function pack(data) {
  const p = data.post || {};
  const base = {
    v: 1, u: data.url || "", at: Date.now(),
    p: [p.id || "", p.handle || "", p.name || "", p.text || "", p.lang || "", p.time || "",
        p.likes || 0, p.replies || 0, p.reposts || 0],
    s: data.selfReplies || 0, m: data.mediaOnly || 0, lo: data.loggedOut ? 1 : 0,
  };
  if (data.error) base.err = data.error;
  const replies = data.replies || [];
  let keep = replies.length;
  for (;;) {   // X lists replies by relevance, so trim from the end until the link is short enough
    const token = await deflateBase64({
      ...base, t: replies.length - keep,
      r: replies.slice(0, keep).map((r) => [r.id, r.handle, r.text, r.likes || 0, r.lang || ""]),
    });
    if (token.length <= MAX_TOKEN || keep === 0) return token;
    keep = Math.floor(keep * 0.85);
  }
}

// ---------- handing results to Senti-Q ----------
async function deliver(data, xTab, origin) {
  const app = await appBase();
  const postUrl = data.url || (xTab?.url || "").split("#")[0];
  const target = `${app}/?post=${encodeURIComponent(postUrl)}&r=${await pack(data)}`;
  const postId = (postUrl.match(/\/status\/(\d+)/) || [])[1];

  // One Senti-Q tab: the one that asked (if Senti-Q asked), else one already showing this
  // post, else the Senti-Q tab you used most recently. A new tab only if there is none.
  const appTabs = (await chrome.tabs.query({})).filter((t) => (t.url || t.pendingUrl || "").startsWith(app));
  let appTab = null;
  if (origin === "app" && xTab?.openerTabId != null) appTab = appTabs.find((t) => t.id === xTab.openerTabId) || null;
  if (!appTab && postId) {
    const marker = new RegExp(`/status/${postId}(?!\\d)`);
    appTab = appTabs.find((t) => marker.test(safeDecode(t.url || ""))) || null;
  }
  if (!appTab && appTabs.length) appTab = appTabs.sort((a, b) => (b.lastAccessed || 0) - (a.lastAccessed || 0))[0];

  if (appTab) {
    await chrome.tabs.update(appTab.id, { url: target, active: true });
    await chrome.windows.update(appTab.windowId, { focused: true });
  } else {
    await chrome.tabs.create({ url: target, index: xTab ? xTab.index + 1 : undefined });
  }
  if (origin === "app" && xTab) chrome.tabs.remove(xTab.id).catch(() => {});
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg?.type === "sentiq:reactions") deliver(msg.data, sender.tab, msg.origin || "here");
});

// ---------- starting a read ----------
async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "sentiq:ping" });
  } catch (e) {   // tab was open before the extension was installed or updated
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  }
}

async function flash(tab, text, title) {
  if (!tab) return;
  await chrome.action.setBadgeBackgroundColor({ color: "#0a0a0a", tabId: tab.id });
  await chrome.action.setBadgeText({ text, tabId: tab.id });
  await chrome.action.setTitle({ title, tabId: tab.id });
  setTimeout(() => {
    chrome.action.setBadgeText({ text: "", tabId: tab.id });
    chrome.action.setTitle({ title: "Read the reactions with Senti-Q", tabId: tab.id });
  }, 4000);
}

async function onAction(tab) {
  const url = tab.url || "";
  if (!X_RE.test(url)) return flash(tab, "!", "Open x.com first, then click Senti-Q.");
  await ensureContentScript(tab.id);
  const type = POST_RE.test(url) ? "sentiq:collect" : "sentiq:pick";
  await chrome.tabs.sendMessage(tab.id, { type });
}

chrome.action.onClicked.addListener((tab) => { onAction(tab); });
self.sentiqAction = onAction;   // used by the automated tests

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "sentiq-link", title: "Read the reactions with Senti-Q", contexts: ["link"],
    targetUrlPatterns: ["https://x.com/*/status/*", "https://twitter.com/*/status/*"],
  });
  chrome.contextMenus.create({
    id: "sentiq-page", title: "Read the reactions with Senti-Q", contexts: ["page"],
    documentUrlPatterns: ["https://x.com/*/status/*", "https://twitter.com/*/status/*"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab) return;
  await ensureContentScript(tab.id);
  if (info.menuItemId === "sentiq-link" && info.linkUrl) {
    const clean = (info.linkUrl.match(POST_RE) || [])[0];
    if (clean) await chrome.tabs.sendMessage(tab.id, { type: "sentiq:readLink", url: clean });
  } else if (info.menuItemId === "sentiq-page") {
    await chrome.tabs.sendMessage(tab.id, { type: "sentiq:collect" });
  }
});
