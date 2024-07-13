function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    const chatBox = document.getElementById('chat-box');

    if (userInput.trim() === '') return;

    chatBox.innerHTML += `<div class="user-message">${userInput}</div>`;
    document.getElementById('user-input').value = '';

    fetch('https://chatbot-frontend-po7w.onrender.com', { // Make sure this URL is correct
        method: 'POST',
        headers: {
            'Content-Type': 'application/json', // Ensure the content type is correct
            'X-CSRFToken': getCookie('csrftoken'), // Include CSRF token if necessary
        },
        body: JSON.stringify({ message: userInput }) // Send data as JSON
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<div class="bot-response">${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(error => console.error('Error:', error));
}

// Ensure the send button event listener is properly set
document.getElementById('send-button').addEventListener('click', sendMessage);

// Optionally handle "Enter" key press for better user experience
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        sendMessage();
        e.preventDefault(); // Prevent newline in the textarea
    }
});
