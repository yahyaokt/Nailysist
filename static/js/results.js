document.addEventListener('DOMContentLoaded', function() {
    // Animasi untuk result cards
    const cards = document.querySelectorAll('.result-card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)'; // Mulai sedikit lebih jauh
        setTimeout(() => {
            card.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 250 + 100); // Penundaan dan waktu animasi sedikit lebih lama
    });

    // Animasi untuk probability circles
    const circles = document.querySelectorAll('.prob-circle');
    circles.forEach((circle, index) => {
        circle.style.opacity = '0';
        circle.style.transform = 'scale(0.8)'; // Mulai dari skala lebih kecil
        setTimeout(() => {
            circle.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            circle.style.opacity = '1';
            circle.style.transform = 'scale(1)';
        }, index * 150 + 400); // Penundaan yang sedikit berbeda
        
        // Tambahkan efek hover
        circle.addEventListener('mouseenter', () => {
            circle.style.transform = 'scale(1.1)';
        });
        
        circle.addEventListener('mouseleave', () => {
            circle.style.transform = 'scale(1)';
        });
    });

    // Animasi untuk tombol "Upload Another Image"
    const uploadAnotherBtn = document.querySelector('.upload-another-btn');
    if (uploadAnotherBtn) {
        uploadAnotherBtn.style.opacity = '0';
        uploadAnotherBtn.style.transform = 'translateY(20px)';
        setTimeout(() => {
            uploadAnotherBtn.style.transition = 'opacity 0.5s ease-out, transform 0.5s ease-out';
            uploadAnotherBtn.style.opacity = '1';
            uploadAnotherBtn.style.transform = 'translateY(0)';
        }, cards.length * 250 + 600); // Setelah kartu muncul
    }
});