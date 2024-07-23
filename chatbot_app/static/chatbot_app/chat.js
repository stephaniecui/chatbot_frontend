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

function formatResponse(response) {
    return response.replace(/\n/g, '<br>');
}

async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === '') return;
    
    updateBubble('user-bubble', userInput);
    document.getElementById('user-input').value = '';

    const botBubble = document.getElementById('bot-bubble');
    botBubble.querySelector('.message-bubble').innerHTML = '';
    botBubble.style.display = 'flex'; // Make bot bubble visible

    try {
        const response = await fetch('/chatbot/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: JSON.stringify({ message: userInput })
        });

        if (response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let result = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                result += decoder.decode(value, { stream: true });
                updateBubble('bot-bubble', formatResponse(result));
            }
        }
    } catch (error) {
        console.error('Error:', error);
        updateBubble('bot-bubble', 'Sorry, an error occurred while processing your request.');
    }
}

function updateBubble(bubbleId, message) {
    const bubble = document.getElementById(bubbleId);
    bubble.querySelector('.message-bubble').innerHTML = message;
    bubble.style.display = 'flex'; // Make sure the bubble is visible
    bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

// Initialize the chat interface
function initChat() {
    const chatBox = document.getElementById('chat-box');
    
    // Create user bubble
    const userBubble = document.createElement('div');
    userBubble.id = 'user-bubble';
    userBubble.className = 'message user-message';
    userBubble.innerHTML = '<div class="message-bubble"></div>';
    userBubble.style.display = 'none'; // Initially hidden
    chatBox.appendChild(userBubble);
    
    // Create bot bubble
    const botBubble = document.createElement('div');
    botBubble.id = 'bot-bubble';
    botBubble.className = 'message bot-message';
    botBubble.innerHTML = '<div class="message-bubble"></div>';
    botBubble.style.display = 'none'; // Initially hidden
    chatBox.appendChild(botBubble);
}

// Ensure the send button event listener is properly set
document.getElementById('send-button').addEventListener('click', sendMessage);

// Optionally handle "Enter" key press for better user experience
document.getElementById('user-input').addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        sendMessage();
        e.preventDefault(); // Prevent newline in the textarea
    }
});

// Initialize the chat interface when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initChat);
