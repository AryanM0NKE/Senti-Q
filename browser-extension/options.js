const input = document.getElementById("appUrl");
const saved = document.getElementById("saved");

chrome.storage.sync.get({ appUrl: "http://localhost:8501" }, ({ appUrl }) => {
  input.value = appUrl;
});

document.getElementById("save").addEventListener("click", () => {
  const value = (input.value || "").trim() || "http://localhost:8501";
  if (!/^https?:\/\//.test(value)) {
    saved.textContent = "Must start with http:// or https://";
    return;
  }
  chrome.storage.sync.set({ appUrl: value }, () => {
    saved.textContent = "Saved";
    setTimeout(() => (saved.textContent = ""), 2000);
  });
});
