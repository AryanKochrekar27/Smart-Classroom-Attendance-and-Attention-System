// static/js/main.js
// small helpers used across pages

console.log("Attendance system static loaded");

// helper to POST JSON
async function postJSON(url, obj) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj),
  });
  return r.json();
}
