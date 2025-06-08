const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');

function triggerFileInput() {
    fileInput.click();
}

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        fileInput.files = e.dataTransfer.files;
        // Update text with file name, make it more prominent
        uploadArea.querySelector('p:first-of-type').textContent = `Gambar dipilih: ${file.name}`;
        uploadArea.querySelector('.or').style.display = 'none'; // Hide "Or"
        uploadArea.querySelector('.select-file-btn').style.display = 'none'; // Hide button
        previewImage(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        // Update text with file name, make it more prominent
        uploadArea.querySelector('p:first-of-type').textContent = `Gambar dipilih: ${file.name}`;
        uploadArea.querySelector('.or').style.display = 'none'; // Hide "Or"
        uploadArea.querySelector('.select-file-btn').style.display = 'none'; // Hide button
        previewImage(file);
    }
});

function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        imagePreview.src = event.target.result;
        imagePreview.style.display = 'block'; // Tampilkan gambar
        imagePreview.style.animation = 'fadeIn 0.5s ease-in-out'; // Animasi fade-in
    }
    reader.readAsDataURL(file);
}