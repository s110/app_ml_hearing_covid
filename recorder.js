/* Grabación y envío de audio con MediaRecorder */
let mediaRecorder, audioChunks = [];
const btnRecord = document.getElementById("btnRecord");
const btnStop   = document.getElementById("btnStop");
const fileInput = document.getElementById("fileInput");
const progBar   = document.getElementById("progBar");
const resultBox = document.getElementById("result");
const form      = document.getElementById("audioForm");

btnRecord.onclick = async () => {
  if (!navigator.mediaDevices) return alert("getUserMedia no soportado");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  audioChunks = [];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.start();
  btnRecord.disabled = true; btnStop.disabled = false;
};

btnStop.onclick = () => {
  mediaRecorder.stop();
  btnRecord.disabled = false; btnStop.disabled = true;
  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: "audio/webm" });
    const file = new File([blob], "recording.webm", { type: blob.type });
    fileInput.files = new DataTransfer().files; // Limpia selección anterior
    // Crea un DataTransfer para insertar el blob como archivo "virtual"
    const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files;
    console.log("🎤 Grabación lista:", file);
  };
};

form.onsubmit = async e => {
  e.preventDefault();
  const sexVal = document.getElementById("sex").value;
  const file   = fileInput.files[0];
  if (!file) return alert("Selecciona o graba un audio primero.");

  progBar.hidden = false; progBar.value = 0;
  const data = new FormData();
  data.append("sex", sexVal);
  data.append("audiofile", file);

  try {
    const resp = await fetch("/upload", {
      method: "POST",
      body: data
    });
    if (!resp.ok) throw new Error(`Error ${resp.status}`);
    const json = await resp.json();
    resultBox.textContent = JSON.stringify(json, null, 2);
  } catch (err) {
    alert("❌ Falló la petición: " + err);
  } finally {
    progBar.hidden = true;
  }
};
