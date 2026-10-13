(() => {
  const input = document.getElementById('leafImage');
  const dropZone = document.getElementById('dropZone');
  const form = document.getElementById('uploadForm');
  const previewWrap = document.getElementById('previewWrap');
  const imagePreview = document.getElementById('imagePreview');
  const uploadPlaceholder = document.getElementById('uploadPlaceholder');
  const fileName = document.getElementById('fileName');
  const analyzeButton = document.getElementById('analyzeButton');
  const buttonText = analyzeButton.querySelector('.button-text');
  const buttonLoading = analyzeButton.querySelector('.button-loading');
  const errorMessage = document.getElementById('errorMessage');
  const emptyResult = document.getElementById('emptyResult');
  const resultContent = document.getElementById('resultContent');

  let selectedFile = null;
  let previewUrl = null;

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.hidden = false;
  }

  function clearError() {
    errorMessage.textContent = '';
    errorMessage.hidden = true;
  }

  function setBusy(busy) {
    analyzeButton.disabled = busy || !selectedFile || analyzeButton.dataset.modelReady === 'false';
    buttonText.hidden = busy;
    buttonLoading.hidden = !busy;
  }

  function selectFile(file) {
    clearError();
    if (!file) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!allowed.includes(file.type)) {
      showError('Please choose a JPG, PNG, WEBP, or BMP image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError('This image is larger than 10 MB. Please resize it and try again.');
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    imagePreview.alt = `Preview of ${file.name}`;
    fileName.textContent = file.name;
    uploadPlaceholder.hidden = true;
    previewWrap.hidden = false;
    if (analyzeButton.dataset.modelReady !== 'false') analyzeButton.disabled = false;
  }

  input.addEventListener('change', (event) => selectFile(event.target.files[0]));
  dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      input.click();
    }
  });
  ['dragenter', 'dragover'].forEach(type => dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach(type => dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.remove('dragging');
  }));
  dropZone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    selectFile(file);
  });

  function renderTopPredictions(predictions) {
    const container = document.getElementById('topPredictions');
    container.replaceChildren();
    predictions.forEach((item, index) => {
      const row = document.createElement('div');
      row.className = 'match-row';
      const rank = document.createElement('span');
      rank.className = 'match-rank';
      rank.textContent = String(index + 1).padStart(2, '0');
      const label = document.createElement('span');
      label.className = 'match-name';
      label.textContent = item.label;
      const score = document.createElement('span');
      score.className = 'match-score';
      score.textContent = `${Number(item.confidence).toFixed(1)}%`;
      row.append(rank, label, score);
      container.append(row);
    });
  }

  function renderAdvice(advice) {
    const list = document.getElementById('adviceList');
    list.replaceChildren();
    advice.forEach(text => {
      const item = document.createElement('li');
      item.textContent = text;
      list.append(item);
    });
  }

  function showResult(data) {
    emptyResult.hidden = true;
    resultContent.hidden = false;
    const healthy = String(data.disease).toLowerCase().includes('healthy');
    const chip = document.getElementById('statusChip');
    chip.textContent = healthy ? 'Healthy-like match' : 'Disease-like match';
    chip.className = `status-chip ${healthy ? 'healthy-chip' : 'disease-chip'}`;
    document.getElementById('predictionTitle').textContent = data.prediction;
    document.getElementById('predictionSubtitle').textContent = `${data.crop} · ${data.disease}`;
    document.getElementById('confidenceLabel').textContent = `${Number(data.confidence).toFixed(1)}% model score`;
    document.getElementById('confidenceBar').style.width = `${Math.max(0, Math.min(100, Number(data.confidence)))}%`;
    renderTopPredictions(data.top_predictions || []);
    document.getElementById('adviceIntro').textContent = data.advice_intro || '';
    renderAdvice(data.advice || []);
    document.getElementById('disclaimer').textContent = data.disclaimer || '';
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    if (!selectedFile) {
      showError('Choose a leaf image first.');
      return;
    }
    if (analyzeButton.dataset.modelReady === 'false') {
      showError('The model has not been trained yet. Follow the training instructions in README.md.');
      return;
    }
    const formData = new FormData();
    formData.append('leaf_image', selectedFile);
    setBusy(true);
    try {
      const response = await fetch('/predict', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed. Please try again.');
      showResult(data);
    } catch (error) {
      showError(error.message || 'Could not contact the server. Make sure the Flask app is running.');
    } finally {
      setBusy(false);
    }
  });

  // A server-rendered flag keeps the frontend honest before a model is trained.
  analyzeButton.dataset.modelReady = analyzeButton.disabled ? 'false' : 'true';
  setBusy(false);
})();
