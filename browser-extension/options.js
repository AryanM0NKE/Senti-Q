const DEFAULTS = { appUrl: "http://localhost:8501", maxReplies: 100 };
const appUrl = document.getElementById("appUrl");
const maxReplies = document.getElementById("maxReplies");
const saved = document.getElementById("saved");

chrome.storage.sync.get(DEFAULTS, (s) => {
  appUrl.value = s.appUrl;
  maxReplies.value = String(s.maxReplies);
});

document.getElementById("save").addEventListener("click", () => {
  const url = (appUrl.value || "").trim() || DEFAULTS.appUrl;
  if (!/^https?:\/\//.test(url)) {
    saved.textContent = "The address must start with http:// or https://";
    return;
  }
  chrome.storage.sync.set({ appUrl: url, maxReplies: parseInt(maxReplies.value, 10) || 100 }, () => {
    saved.textContent = "Saved";
    setTimeout(() => (saved.textContent = ""), 2000);
  });
});
