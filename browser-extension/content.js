// Senti-Q for X — content script (x.com / twitter.com pages).
//
// Reads the replies that X shows on a post's page, but only when you ask:
//   * Senti-Q's "Read the replies on X" button (the link ends in #senti-q),
//   * the toolbar icon on a post's page (reads that post right there), or
//   * the toolbar icon anywhere else: you pick a post, it's opened in this tab, read,
//     and you're taken back to where you were.
// It reads the post and, for each reply, its text, @handle, like count and language.
// It never reads your account, messages or cookies, and sends nothing over the
// network: the results go to the extension, which opens Senti-Q with them.

(() => {
  if (window.__sentiqLoaded) return;
  window.__sentiqLoaded = true;

  const MARK = "senti-q";
  const PENDING = "senti-q:pending";
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

  const timeLink = (article) => {
    const links = [...article.querySelectorAll('a[href*="/status/"]')];
    return links.find((a) => a.querySelector("time")) || null;
  };

  function parse(article) {
    const link = timeLink(article);
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
  const xIsLoading = () => !!document.querySelector('[role="progressbar"]');
  const atBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 8;

  // ---------- small pill / banner on the X page ----------
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
    let quiet = 0;          // passes in a row that found nothing new while X wasn't loading
    let waits = 0;          // consecutive waits for X's loading spinner (capped, in case one never goes away)

    while (replies.size < maxReplies && Date.now() - started < 75000) {
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
      if (replies.size >= maxReplies) break;

      // Reveal replies X hides behind buttons (often the hostile ones)
      let revealed = false;
      for (const b of document.querySelectorAll('[role="button"], button')) {
        const label = (b.textContent || "").trim();
        if (!clicked.has(b) && /^(show (more|additional) replies|show probable spam)/i.test(label) && (!cut || !follows(cut, b))) {
          clicked.add(b);
          b.click();
          revealed = true;
          await sleep(900);
        }
      }

      const found = replies.size + selfReplies.size + mediaOnly.size - before;
      if (found > 0) waits = 0;
      if (xIsLoading() && waits < 6) {          // X is still fetching replies: wait, don't give up
        waits += 1;
        await sleep(500);
        continue;
      }
      waits = 0;
      quiet = found > 0 || revealed ? 0 : quiet + 1;
      const end = (cut && cut.getBoundingClientRect().top < window.innerHeight) || atBottom();
      if (end && quiet >= 3) break;             // the end of the conversation, and nothing more loaded
      if (quiet >= 10) break;
      window.scrollBy(0, Math.round(window.innerHeight * 0.8));
      await sleep(end ? 1000 : 700);
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

  // origin: "app"  - Senti-Q asked (#senti-q link): Senti-Q closes this tab afterwards
  //         "here" - toolbar click on this post's page: stay here
  //         "pick" - you picked a post elsewhere: go back to where you were afterwards
  async function run(origin) {
    if (running) return;
    running = true;
    try {
      const { maxReplies } = await chrome.storage.sync.get({ maxReplies: 100 });
      const data = await collect(Math.min(Math.max(parseInt(maxReplies, 10) || 100, 10), 300));
      show(data.error ? "Senti-Q · couldn't read this post" : `Senti-Q · read ${data.replies.length} replies, opening Senti-Q…`);
      await chrome.runtime.sendMessage({ type: "sentiq:reactions", data, origin });
      if (origin === "pick") setTimeout(() => history.back(), 600);
    } finally {
      running = false;
      setTimeout(hide, 2500);
    }
  }

  // ---------- picking a post (toolbar click on a timeline, search, profile…) ----------
  let picker = null;

  function readPicked(article) {
    const p = parse(article);
    if (!p) return;
    sessionStorage.setItem(PENDING, JSON.stringify({ id: p.id, at: Date.now() }));
    pendingId = p.id;
    const link = timeLink(article);
    if (link) link.click();                       // X opens the post in this tab
    else location.assign(canonical(p.id, p.handle));
  }

  function startPicking() {
    if (picker || running) return;
    const banner = document.createElement("div");
    banner.textContent = "Senti-Q · click the post you want to read   (Esc to cancel)";
    banner.setAttribute("style",
      "position:fixed;z-index:2147483647;left:50%;top:16px;transform:translateX(-50%);padding:12px 20px;" +
      "border-radius:999px;background:#0a0a0a;color:#fff;font:500 14px/1.2 system-ui,-apple-system,'Segoe UI',sans-serif;" +
      "box-shadow:0 8px 30px rgba(0,0,0,.25);pointer-events:none");
    document.documentElement.appendChild(banner);

    let hovered = null;
    const mark = (a) => {
      if (hovered === a) return;
      if (hovered) { hovered.style.outline = ""; hovered.style.cursor = ""; }
      hovered = a;
      if (a) { a.style.outline = "3px solid #0a0a0a"; a.style.outlineOffset = "-3px"; a.style.cursor = "pointer"; }
    };
    const over = (e) => mark(e.target.closest ? e.target.closest("article") : null);
    const click = (e) => {
      const a = e.target.closest ? e.target.closest("article") : null;
      if (!a) return;
      e.preventDefault();
      e.stopPropagation();
      stop();
      readPicked(a);
    };
    const key = (e) => { if (e.key === "Escape") stop(); };
    const timer = setTimeout(() => stop(), 60000);

    function stop() {
      clearTimeout(timer);
      document.removeEventListener("mouseover", over, true);
      document.removeEventListener("click", click, true);
      document.removeEventListener("keydown", key, true);
      mark(null);
      banner.remove();
      picker = null;
    }
    document.addEventListener("mouseover", over, true);
    document.addEventListener("click", click, true);
    document.addEventListener("keydown", key, true);
    picker = { stop };
  }

  // A picked post is read once its page is showing (X switches pages without reloading,
  // so watch the address; after a full reload, the pending pick is in sessionStorage).
  let pendingId = null;
  try {
    const p = JSON.parse(sessionStorage.getItem(PENDING) || "null");
    if (p && Date.now() - p.at < 60000) pendingId = p.id;
    else sessionStorage.removeItem(PENDING);
  } catch (e) { sessionStorage.removeItem(PENDING); }

  setInterval(() => {
    if (pendingId && !running && statusId(location.pathname) === pendingId) {
      pendingId = null;
      sessionStorage.removeItem(PENDING);
      run("pick");
    }
  }, 300);

  chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
    if (msg?.type === "sentiq:ping") reply({ ok: true });
    else if (msg?.type === "sentiq:collect") { run("here"); reply({ ok: true }); }
    else if (msg?.type === "sentiq:pick") { startPicking(); reply({ ok: true }); }
    else if (msg?.type === "sentiq:readLink") {
      const id = statusId(msg.url);
      const article = id && articles().find((a) => parse(a)?.id === id);
      if (article) readPicked(article);
      else if (id) {
        sessionStorage.setItem(PENDING, JSON.stringify({ id, at: Date.now() }));
        pendingId = id;
        location.assign(msg.url);
      }
      reply({ ok: true });
    }
  });

  if (location.hash.includes(MARK)) {                   // opened by Senti-Q
    const go = () => run("app");
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go, { once: true });
    else go();
  }
})();
