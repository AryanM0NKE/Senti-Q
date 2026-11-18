// Senti-Q for X — hands the post you're looking at on X to the Senti-Q web app.
//
// * Toolbar icon: on a post's own page, analyses that post. On a timeline, analyses
//   the post nearest the middle of your screen.
// * Right-click any post link (e.g. its timestamp) -> "Analyse with Senti-Q".
//
// The extension never reads your X account, cookies or password. It only reads the
// link of one post, and only when you click, then opens Senti-Q with that link.

const DEFAULT_APP_URL = "http://localhost:8501";
const POST_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\/(?:i\/web|[^/]+)\/status\/\d+/;
const X_PAGE_RE = /^https:\/\/(?:www\.)?(?:x|twitter)\.com\//;

async function appUrl() {
  const { appUrl } = await chrome.storage.sync.get({ appUrl: DEFAULT_APP_URL });
  return appUrl.replace(/\/+$/, "");
}

function cleanPostUrl(url) {
  const m = url.match(POST_RE);
  return m ? m[0] : null;
}

async function openInSentiQ(postUrl, tab) {
  const target = `${await appUrl()}/?post=${encodeURIComponent(postUrl)}`;
  await chrome.tabs.create({ url: target, index: tab ? tab.index + 1 : undefined });
}

async function flash(tab, text, title) {
  if (!tab) return;
  await chrome.action.setBadgeBackgroundColor({ color: "#0a0a0a", tabId: tab.id });
  await chrome.action.setBadgeText({ text, tabId: tab.id });
  await chrome.action.setTitle({ title, tabId: tab.id });
  setTimeout(() => {
    chrome.action.setBadgeText({ text: "", tabId: tab.id });
    chrome.action.setTitle({ title: "Analyse this post with Senti-Q", tabId: tab.id });
  }, 4000);
}

// Runs inside the X page: link of the post closest to the vertical centre of the screen.
function findCentredPost() {
  const mid = window.innerHeight / 2;
  let best = null;
  let bestDist = Infinity;
  for (const article of document.querySelectorAll("article")) {
    const r = article.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight) continue;
    const links = [...article.querySelectorAll('a[href*="/status/"]')];
    const link = links.find((a) => a.querySelector("time")) || links[0];
    if (!link) continue;
    const dist = Math.abs(r.top + r.height / 2 - mid);
    if (dist < bestDist) {
      bestDist = dist;
      best = link.href;
    }
  }
  return best;
}

chrome.action.onClicked.addListener(async (tab) => {
  const url = tab.url || "";
  if (!X_PAGE_RE.test(url)) {
    return flash(tab, "!", "Open x.com first, then click to analyse a post.");
  }
  let postUrl = cleanPostUrl(url);
  if (!postUrl) {
    try {
      const [{ result }] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: findCentredPost,
      });
      postUrl = result ? cleanPostUrl(result) : null;
    } catch (e) {
      postUrl = null;
    }
  }
  if (!postUrl) {
    return flash(tab, "?", "No post found here. Scroll a post into view, or open it.");
  }
  await openInSentiQ(postUrl, tab);
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "sentiq-link",
    title: "Analyse with Senti-Q",
    contexts: ["link"],
    targetUrlPatterns: ["https://x.com/*/status/*", "https://twitter.com/*/status/*"],
  });
  chrome.contextMenus.create({
    id: "sentiq-page",
    title: "Analyse this post with Senti-Q",
    contexts: ["page"],
    documentUrlPatterns: ["https://x.com/*/status/*", "https://twitter.com/*/status/*"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const postUrl = cleanPostUrl(info.linkUrl || info.pageUrl || "");
  if (postUrl) await openInSentiQ(postUrl, tab);
});
