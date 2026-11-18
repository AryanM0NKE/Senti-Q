// Senti-Q for X — content script (x.com / twitter.com pages).
//
// Reads the replies that X shows on a post's page, but only when asked:
//   * the page was opened by Senti-Q (its link ends in #senti-q), or
//   * you clicked the Senti-Q toolbar icon / context-menu item on this page.
// It reads the post and, for each reply, its text, @handle, like count and language.
// It never reads your account, messages or cookies, and sends nothing over the
// network: the results go to the extension, which opens Senti-Q with them.

(() => {
  if (window.__sentiqLoaded) return;
  window.__sentiqLoaded = true;

  const MARK = "senti-q";
  const openedBySentiQ = location.hash.includes(MARK);
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
  const statusId = (href) => ((href || "").match(/\/status\/(\d+)/) || [])[1] || null;
  const canonical = (id, handle) => `https://x.com/${handle || "i"}/status/${id}`;
  let running = false;

  // ---------- reading one post (an <article>) ----------
  function textOf(node) {
    let out = "";
    for (const n of node.childNodes) {
      if (n.nodeType === Node.TEXT_NODE) out += n.nodeValue;
      else if (n.nodeName === "IMG") out += n.getAttribute("alt") || "";   // X draws emoji as images
      else if (n.nodeName === "BR") out += "\n";
      else if (n.nodeType === Node.ELEMENT_NODE) out += textOf(n);
    }
    return out;
  }

  function count(article, testIds) {
    for (const id of testIds) {
      const el = article.querySelector(`[data-testid="${id}"]`);
      if (el) {
        const m = (el.getAttribute("aria-label") || "").replace(/,/g, "").match(/\d+/);
        return m ? parseInt(m[0], 10) : 0;
      }
    }
    return 0;
  }

  function parse(article) {
    const links = [...article.querySelectorAll('a[href*="/status/"]')];
    const link = links.find((a) => a.querySelector("time"));
    const id = link ? statusId(link.getAttribute("href")) : null;
    if (!id) return null;

    let handle = "";
    let name = "";
    const user = article.querySelector('[data-testid="User-Name"]');
    if (user) {
      const href = [...user.querySelectorAll('a[href^="/"]')]
        .map((a) => a.getAttribute("href"))
        .find((h) => /^\/[A-Za-z0-9_]{1,30}$/.test(h));
      handle = href ? href.slice(1) : "";
      const first = user.querySelector("span");
      name = first ? textOf(first).trim() : "";
    }
    if (!handle) handle = link.getAttribute("href").split("/")[1] || "";

    const body = article.querySelector('[data-testid="tweetText"]');
    const spans = [...article.querySelectorAll("span")].map((s) => (s.textContent || "").trim());
    return {
      id, handle, name,
      text: body ? textOf(body).trim() : "",
      lang: body ? (body.getAttribute("lang") || "") : "",
      time: article.querySelector("time")?.getAttribute("datetime") || "",
      likes: count(article, ["like", "unlike"]),
      replies: count(article, ["reply"]),
      reposts: count(article, ["retweet", "unretweet"]),
      isAd: !!article.querySelector('[data-testid="placementTracking"]') || spans.includes("Ad") || spans.includes("Promoted"),
    };
  }

  const articles = () => [...document.querySelectorAll("article")];
  const follows = (a, b) => !!(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING); // b comes after a
  const discoverHeading = () =>
    [...document.querySelectorAll('h2, [role="heading"]')].find((h) => /^\s*discover more\s*$/i.test(h.textContent || "")) || null;
  const signInWall = () =>
    !!document.querySelector('[data-testid="loginButton"], [data-testid="LoginForm_Login_Button"], a[href="/login"], a[href="/i/flow/login"]');

  // ---------- small progress pill on the X page ----------
  let pill = null;
  function show(text) {
    if (!pill) {
      pill = document.createElement("div");
      pill.setAttribute("style",
        "position:fixed;z-index:2147483647;right:20px;bottom:20px;padding:12px 18px;border-radius:999px;" +
        "background:#0a0a0a;color:#fff;font:500 14px/1.2 system-ui,-apple-system,'Segoe UI',sans-serif;" +
        "box-shadow:0 8px 30px rgba(0,0,0,.25);pointer-events:none");
      document.documentElement.appendChild(pill);
    }
    pill.textContent = text;
  }
  const hide = () => { pill?.remove(); pill = null; };

  // ---------- collecting the replies ----------
  async function collect(maxReplies) {
    const focalId = statusId(location.pathname);
    if (!focalId) return { error: "notfound", url: location.href.split("#")[0] };
    show("Senti-Q · opening the post…");

    let focal = null;
    for (let i = 0; i < 60 && !focal; i++) {                 // wait up to ~18 s for X to render
      focal = articles().find((a) => parse(a)?.id === focalId) || null;
      if (!focal) await sleep(300);
    }
    if (!focal) return { error: signInWall() ? "login" : "notfound", url: canonical(focalId) };

    const post = parse(focal);
    const ancestors = new Set();                             // the thread above the post
    for (const a of articles()) {
      if (a === focal) break;
      const p = parse(a);
      if (p) ancestors.add(p.id);
    }

    const replies = new Map();
    const selfReplies = new Set();
    const mediaOnly = new Set();
    const clicked = new WeakSet();
    const author = (post.handle || "").toLowerCase();
    const startY = window.scrollY;
    const started = Date.now();
    let idle = 0;

    while (replies.size < maxReplies && idle < 8 && Date.now() - started < 60000) {
      const before = replies.size + selfReplies.size + mediaOnly.size;
      const cut = discoverHeading();

      for (const a of articles()) {
        if (a === focal) continue;
        if (document.contains(focal) && !follows(focal, a)) continue;   // above the post
        if (cut && follows(cut, a)) continue;                          // X's "Discover more" posts
        const p = parse(a);
        if (!p || p.id === focalId || ancestors.has(p.id) || p.isAd) continue;
        if (replies.has(p.id) || selfReplies.has(p.id) || mediaOnly.has(p.id)) continue;
        if (author && p.handle.toLowerCase() === author) { selfReplies.add(p.id); continue; }
        if (!p.text) { mediaOnly.add(p.id); continue; }
        replies.set(p.id, p);
        if (replies.size >= maxReplies) break;
      }
      show(`Senti-Q · reading replies… ${replies.size}`);

      // Reveal replies X hides behind buttons (often the hostile ones)
      for (const b of document.querySelectorAll('[role="button"], button')) {
        const label = (b.textContent || "").trim();
        if (!clicked.has(b) && /^(show (more|additional) replies|show probable spam)/i.test(label) && (!cut || !follows(cut, b))) {
          clicked.add(b);
          b.click();
          await sleep(900);
        }
      }

      const found = replies.size + selfReplies.size + mediaOnly.size - before;
      if (cut && cut.getBoundingClientRect().top < window.innerHeight && found === 0) break;   // end of the conversation
      window.scrollBy(0, Math.round(window.innerHeight * 0.85));
      await sleep(700);
      idle = found === 0 ? idle + 1 : 0;
    }
    window.scrollTo(0, startY);

    return {
      url: canonical(focalId, post.handle),
      post: { id: post.id, handle: post.handle, name: post.name, text: post.text, lang: post.lang,
              time: post.time, likes: post.likes, replies: post.replies, reposts: post.reposts },
      replies: [...replies.values()].slice(0, maxReplies)
        .map((r) => ({ id: r.id, handle: r.handle, text: r.text, likes: r.likes, lang: r.lang })),
      selfReplies: selfReplies.size,
      mediaOnly: mediaOnly.size,
      loggedOut: replies.size === 0 && signInWall(),
    };
  }

  // Post nearest the middle of the screen (for clicks on a timeline)
  function centredPost() {
    const mid = window.innerHeight / 2;
    let best = null;
    let bestDist = Infinity;
    for (const a of articles()) {
      const r = a.getBoundingClientRect();
      if (r.bottom < 0 || r.top > window.innerHeight) continue;
      const p = parse(a);
      if (!p) continue;
      const dist = Math.abs(r.top + r.height / 2 - mid);
      if (dist < bestDist) { bestDist = dist; best = canonical(p.id, p.handle); }
    }
    return best;
  }

  async function run(openedByApp) {
    if (running) return;
    running = true;
    try {
      const { maxReplies } = await chrome.storage.sync.get({ maxReplies: 100 });
      const data = await collect(Math.min(Math.max(parseInt(maxReplies, 10) || 100, 10), 300));
      show(data.error ? "Senti-Q · couldn't read this post" : `Senti-Q · read ${data.replies.length} replies, opening Senti-Q…`);
      await chrome.runtime.sendMessage({ type: "sentiq:reactions", data, openedByApp });
    } finally {
      running = false;
      setTimeout(hide, 2500);
    }
  }

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg?.type === "sentiq:ping") reply({ ok: true });
    else if (msg?.type === "sentiq:collect") { run(false); reply({ ok: true }); }
    else if (msg?.type === "sentiq:centred") reply({ url: centredPost() });
  });

  if (openedBySentiQ) {
    const go = () => run(true);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go, { once: true });
    else go();
  }
})();
