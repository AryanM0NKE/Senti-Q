// Senti-Q for X — background service worker.
//
// * Toolbar icon on a post's page: reads that post's replies, then opens Senti-Q.
// * Toolbar icon on a timeline: opens the post in the middle of your screen, reads its
//   replies, closes that tab and opens Senti-Q.
// * Right-click a post's link -> "Read the reactions with Senti-Q".
// * Links from Senti-Q ("Read the replies on X") end in #senti-q: the replies are read
//   automatically and you're taken back to the Senti-Q tab you came from.
//
// The replies travel inside the Senti-Q link itself (compressed), so nothing is stored
// or sent anywhere else.

const DEFAULTS = { appUrl: "http://localhost:8501", maxReplies: 100 };
const MARK = "senti-q";
const MAX_TOKEN = 24000;   // characters of packed replies in the Senti-Q link
const POST_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\/(?:i\/web|[^/?#]+)\/status\/(\d+)/;
const X_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\//;

async function appBase() {
  const { appUrl } = await chrome.storage.sync.get(DEFAULTS);
  return appUrl.replace(/\/+$/, "");
}

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
async function deliver(data, xTab, openedByApp) {
  const app = await appBase();
  const postUrl = data.url || (xTab?.url || "").split("#")[0];
  const target = `${app}/?post=${encodeURIComponent(postUrl)}&r=${await pack(data)}`;
  const postId = (postUrl.match(/\/status\/(\d+)/) || [])[1];

  // When Senti-Q asked for the replies ("Read the replies on X"), return to that Senti-Q tab:
  // the one this X tab was opened from, else any Senti-Q tab showing this post. A toolbar
  // click always opens a fresh Senti-Q tab, so an earlier analysis is never replaced.
  let appTab = null;
  if (openedByApp && xTab?.openerTabId != null) {
    try {
      const t = await chrome.tabs.get(xTab.openerTabId);
      if ((t.url || t.pendingUrl || "").startsWith(app)) appTab = t;
    } catch (e) { /* opener closed */ }
  }
  if (!appTab && openedByApp && postId) {
    const tabs = await chrome.tabs.query({});
    const marker = new RegExp(`/status/${postId}(?!\\d)`);
    appTab = tabs.find((t) => (t.url || "").startsWith(app) && marker.test(decodeURIComponent(t.url))) || null;
  }

  if (appTab) {
    await chrome.tabs.update(appTab.id, { url: target, active: true });
    await chrome.windows.update(appTab.windowId, { focused: true });
  } else {
    await chrome.tabs.create({ url: target, index: xTab ? xTab.index + 1 : undefined });
  }
  if (openedByApp && xTab) chrome.tabs.remove(xTab.id).catch(() => {});
}

chrome.runtime.onMessage.addListener((msg, sender) => {
  if (msg?.type === "sentiq:reactions") deliver(msg.data, sender.tab, !!msg.openedByApp);
});

// ---------- starting a read ----------
async function ensureContentScript(tabId) {
  try {
    await chrome.tabs.sendMessage(tabId, { type: "sentiq:ping" });
  } catch (e) {   // tab was open before the extension was installed
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
  }
}

async function openAndRead(postUrl, fromTab) {
  const clean = (postUrl.match(POST_RE) || [])[0];
  if (!clean) return;
  await chrome.tabs.create({ url: `${clean}#${MARK}`, active: true,
                             index: fromTab ? fromTab.index + 1 : undefined, openerTabId: fromTab?.id });
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
  if (!X_RE.test(url)) return flash(tab, "!", "Open a post on x.com, then click Senti-Q.");
  await ensureContentScript(tab.id);
  if (POST_RE.test(url)) {
    await chrome.tabs.sendMessage(tab.id, { type: "sentiq:collect" });
    return;
  }
  const res = await chrome.tabs.sendMessage(tab.id, { type: "sentiq:centred" }).catch(() => null);
  if (!res?.url) return flash(tab, "?", "No post found here. Scroll a post into view, or open it.");
  await openAndRead(res.url, tab);
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
  if (info.menuItemId === "sentiq-link" && info.linkUrl) return openAndRead(info.linkUrl, tab);
  if (info.menuItemId === "sentiq-page" && tab) {
    await ensureContentScript(tab.id);
    await chrome.tabs.sendMessage(tab.id, { type: "sentiq:collect" });
  }
});
