document.addEventListener('DOMContentLoaded', function() {
    function sendMessage() {
        const userInput = document.getElementById('user-input').value;
        const chatBox = document.getElementById('chat-box');

        if (userInput.trim() === '') return;

        chatBox.innerHTML += `<div class="user-message">${userInput}</div>`;
        document.getElementById('user-input').value = '';

        fetch('https://your-backend-api-url/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: userInput })
        })
        .then(response => response.json())
        .then(data => {
            chatBox.innerHTML += `<div class="bot-response">${data.response}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        })
        .catch(error => console.error('Error:', error));
    }

    document.getElementById('send-button').addEventListener('click', sendMessage);
});

