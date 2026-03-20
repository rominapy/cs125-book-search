
function toggleDetails(id) {
    const el = document.getElementById(id);
    if (el.style.display === "none") {
        el.style.display = "block";
    } else {
        el.style.display = "none";
    }
}

function clearPrefs() {
  document.getElementById("mood-input").value = "";
  document.getElementById("time-input").value = "";
  fetch('/clear_prefs', { method: 'POST' });

}
