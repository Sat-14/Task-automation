# Multi-Agent Task Automation System

A Python-based system that allows users to describe tasks in natural language and have them executed by a team of specialized agents. It uses Google's Gemini API for natural language understanding and orchestrates various agents to complete complex tasks.



## Features

- Natural language task description processing
- Multi-agent system with specialized agents for different tasks
- Real-time execution status updates via WebSockets
- Web-based user interface
- Task history tracking
- Integration with external services (Calendar, Slack, Email, etc.)

## Quick Start

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. Clone the repository or download the app.py file

2. Install the required dependencies:
```bash
pip install fastapi uvicorn requests python-dotenv
```

3. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:8000`

## Usage

1. Enter a task description in natural language in the text box
   - Example: "Find doctors near me who check skin related issues"
   - Example: "Send hi to example@email.com"

2. Click "Execute Task" to submit your task

3. Watch as the system:
   - Creates an execution plan
   - Executes each step in sequence
   - Shows real-time progress and logs

## Available Agents

The system includes the following agents:

- **Knowledge Agent**: Retrieves information from internal knowledge base
- **Search Agent**: Performs web searches for information
- **Slack Agent**: Posts messages to Slack channels
- **Email Agent**: Sends emails to specified recipients
- **Communication Agent**: Makes phone calls or sends text messages
- **Calendar Agent**: Manages calendar events
- **Weather Agent**: Retrieves weather information
- **Booking Agent**: Handles reservations and bookings

## Configuration

You can configure additional API keys for specific agents in the `.env` file:

```
# Required for basic functionality
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: For communication agent (Twilio)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=your_twilio_phone_number

# Optional: For Slack integration
SLACK_BOT_TOKEN=your_slack_bot_token

# Optional: For other integrations
EMAIL_API_KEY=your_email_api_key
WEATHER_API_KEY=your_weather_api_key
BOOKING_API_KEY=your_booking_api_key
```

If these optional keys are not provided, the agents will operate in simulation mode.

## Adding Your Own Knowledge

Place text files with domain-specific knowledge in the `knowledge_base` directory. These files will be accessible to the Knowledge Agent.

## How It Works

1. **Natural Language Understanding**: The system uses Google's Gemini API to parse the natural language task and create an execution plan
2. **Task Orchestration**: The TaskOrchestrator breaks down complex tasks into individual steps assigned to specific agents
3. **Agent Execution**: Each agent executes its specialized function, from searching the web to sending emails
4. **Real-time Updates**: WebSocket connections provide real-time updates on task progress

## Extending the System

To add a new agent:

1. Create a new agent class that inherits from `BaseAgent`
2. Implement the `run()` method with appropriate parameters
3. Add a parser prompt template to handle natural language instructions
4. Register the agent in the `TaskOrchestrator.__init__()` method
5. Add the parser template to the `parsers` dictionary

## Troubleshooting

- **API Key Issues**: Ensure your Gemini API key is correctly set in the `.env` file
- **UI Not Updating**: Check browser console for any JavaScript errors
- **Agent Failures**: Review the logs to identify which agent is failing and why

## License

[MIT License](LICENSE)

## Acknowledgements

- Google Gemini API for natural language processing
- FastAPI for the web framework
- Various service providers for external integrations
