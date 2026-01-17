"""Web interface for Humanize text humanization tool."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import sys

# Handle import for both module and script execution
try:
    from humanize import humanize
except ImportError:
    # When running as a script, add parent directory to path
    parent_dir = Path(__file__).parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from humanize import humanize

app = FastAPI(
    title="Humanize",
    description="An open-source, rule-based system for humanizing text",
    version="1.0.0",
)


class TextRequest(BaseModel):
    """Request model for text humanization."""

    text: str


class TextResponse(BaseModel):
    """Response model for humanized text."""

    original: str
    humanized: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Humanize - Text Humanization Tool</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 900px;
            width: 100%;
            padding: 40px;
        }

        h1 {
            color: #333;
            margin-bottom: 8px;
            font-size: 2.5em;
            font-weight: 700;
        }

        .subtitle {
            color: #666;
            margin-bottom: 32px;
            font-size: 1.1em;
        }

        .textarea-container {
            margin-bottom: 24px;
        }

        label {
            display: block;
            color: #333;
            font-weight: 600;
            margin-bottom: 8px;
            font-size: 0.95em;
        }

        textarea {
            width: 100%;
            min-height: 200px;
            padding: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-family: inherit;
            font-size: 1em;
            resize: vertical;
            transition: border-color 0.3s;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        textarea:disabled {
            background-color: #f5f5f5;
            cursor: not-allowed;
        }

        .button-group {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
        }

        button {
            flex: 1;
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #f5f5f5;
            color: #333;
        }

        .btn-secondary:hover {
            background: #e0e0e0;
        }

        .output-section {
            margin-top: 32px;
            padding-top: 32px;
            border-top: 2px solid #f0f0f0;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }

        .copy-btn {
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 0.9em;
            cursor: pointer;
            transition: background 0.3s;
        }

        .copy-btn:hover {
            background: #5568d3;
        }

        .copy-btn:active {
            transform: scale(0.98);
        }

        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-weight: 600;
            margin: 20px 0;
        }

        .error {
            display: none;
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 16px;
            border: 1px solid #fcc;
        }

        .char-count {
            color: #999;
            font-size: 0.9em;
            margin-top: 8px;
            text-align: right;
        }

        @media (max-width: 600px) {
            .container {
                padding: 24px;
            }

            h1 {
                font-size: 2em;
            }

            .button-group {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Humanize</h1>
        <p class="subtitle">Transform machine-like text into natural, human-readable prose</p>

        <form id="humanize-form">
            <div class="textarea-container">
                <label for="input-text">Enter text to humanize:</label>
                <textarea id="input-text" name="text" placeholder="Paste or type your text here..."></textarea>
                <div class="char-count" id="input-count">0 characters</div>
            </div>

            <div class="button-group">
                <button type="submit" class="btn-primary" id="humanize-btn">Humanize Text</button>
                <button type="button" class="btn-secondary" id="clear-btn">Clear</button>
            </div>

            <div class="error" id="error-message"></div>
            <div class="loading" id="loading">Processing your text...</div>

            <div class="output-section" id="output-section" style="display: none;">
                <div class="output-header">
                    <label>Humanized text:</label>
                    <button type="button" class="copy-btn" id="copy-btn">Copy</button>
                </div>
                <textarea id="output-text" readonly></textarea>
                <div class="char-count" id="output-count">0 characters</div>
            </div>
        </form>
    </div>

    <script>
        const form = document.getElementById('humanize-form');
        const inputText = document.getElementById('input-text');
        const outputText = document.getElementById('output-text');
        const outputSection = document.getElementById('output-section');
        const loading = document.getElementById('loading');
        const errorMsg = document.getElementById('error-message');
        const humanizeBtn = document.getElementById('humanize-btn');
        const clearBtn = document.getElementById('clear-btn');
        const copyBtn = document.getElementById('copy-btn');
        const inputCount = document.getElementById('input-count');
        const outputCount = document.getElementById('output-count');

        // Update character count
        inputText.addEventListener('input', () => {
            inputCount.textContent = inputText.value.length + ' characters';
        });

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const text = inputText.value.trim();
            if (!text) {
                showError('Please enter some text to humanize.');
                return;
            }

            // Hide previous output and errors
            outputSection.style.display = 'none';
            errorMsg.style.display = 'none';
            loading.style.display = 'block';
            humanizeBtn.disabled = true;

            try {
                const response = await fetch('/api/humanize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to humanize text');
                }

                const data = await response.json();
                outputText.value = data.humanized;
                outputCount.textContent = data.humanized.length + ' characters';
                outputSection.style.display = 'block';
            } catch (err) {
                showError(err.message || 'An error occurred while processing your text.');
            } finally {
                loading.style.display = 'none';
                humanizeBtn.disabled = false;
            }
        });

        // Clear button
        clearBtn.addEventListener('click', () => {
            inputText.value = '';
            outputText.value = '';
            outputSection.style.display = 'none';
            errorMsg.style.display = 'none';
            inputCount.textContent = '0 characters';
            outputCount.textContent = '0 characters';
        });

        // Copy button
        copyBtn.addEventListener('click', () => {
            outputText.select();
            document.execCommand('copy');
            
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            setTimeout(() => {
                copyBtn.textContent = originalText;
            }, 2000);
        });

        function showError(message) {
            errorMsg.textContent = message;
            errorMsg.style.display = 'block';
        }
    </script>
</body>
</html>
"""


@app.post("/api/humanize", response_model=TextResponse)
async def humanize_text(request: TextRequest):
    """Humanize the provided text."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        humanized = humanize(request.text)
        return TextResponse(original=request.text, humanized=humanized)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

