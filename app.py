"""
Multi-Agent Task Automation System
---------------------------------
A complete, self-contained application that allows users to describe tasks in natural language
and have them executed by a team of specialized agents.

Run with: python app.py
"""

import os
import json
import logging
import asyncio
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uvicorn
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY", "")
GOOGLE_CALENDAR_CREDENTIALS_FILE = os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_FILE", "")
GOOGLE_CALENDAR_TOKEN_FILE = os.environ.get("GOOGLE_CALENDAR_TOKEN_FILE", "")

if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. System will not function correctly!")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# Initialize FastAPI app
app = FastAPI(title="Multi-Agent Task Automation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory task storage
tasks_db: Dict[str, Dict[str, Any]] = {}
task_counter = 0

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class TaskRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = "anonymous"

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

# =============================================================================
# WEBSOCKET CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections with clients."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new client."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        """Disconnect a client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Remaining connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
            except RuntimeError:
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

# Create a global instance of the connection manager
ws_manager = ConnectionManager()

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are TaskMaster, an advanced AI assistant that helps users accomplish tasks through a team of specialized agents.
Your primary responsibility is to understand the user's request, break it down into actionable steps, and assign each step to the appropriate agent."""

PLANNER_PROMPT_TEMPLATE = """
{system_prompt}

You need to create a detailed execution plan for the following request:
"{user_prompt}"

Available agents:
1. "KnowledgeAgent": Retrieves information from internal knowledge base (use for company-specific questions)
2. "SearchAgent": Performs web searches to find current information
3. "SlackAgent": Posts messages to specific Slack channels
4. "EmailAgent": Sends emails to specified recipients
5. "CommunicationAgent": Makes phone calls or sends text messages
6. "CalendarAgent": Checks, creates, updates or deletes calendar events
7. "WeatherAgent": Gets current weather or forecasts for specific locations
8. "BookingAgent": Books appointments, reservations, tickets, etc.

Create a JSON array of steps needed to complete this task. Each step should include:
- "agent": The name of the agent to use (from the list above)
- "action": A specific instruction for that agent
- "description": A user-friendly description of this step

Think step-by-step. If multiple agents are needed, arrange them in logical order.
If you need to store information from one step to use in another, indicate this in the action.

Example output format:
[
  {{
    "agent": "SearchAgent",
    "action": "Find top-rated Italian restaurants in downtown",
    "description": "Searching for Italian restaurants in the downtown area"
  }},
  {{
    "agent": "BookingAgent",
    "action": "Book a table for 2 at [restaurant_name] for tomorrow at 7pm",
    "description": "Making a reservation for 2 people"
  }},
  {{
    "agent": "CalendarAgent",
    "action": "Add dinner reservation to calendar for tomorrow at 7pm",
    "description": "Adding the reservation to your calendar"
  }}
]

Now create the execution plan:
"""

SLACK_PARSER_PROMPT_TEMPLATE = """
Extract the Slack channel and message content from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "channel": The Slack channel to post to (should start with #)
- "message": The message content to post

JSON Output:
"""

EMAIL_PARSER_PROMPT_TEMPLATE = """
Extract the email details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "to": The recipient's email address
- "subject": The email subject line
- "body": The email body content

JSON Output:
"""

EVENT_PARSER_PROMPT_TEMPLATE = """
Extract the calendar event details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "title": The event title
- "start_time": The start time in format "YYYY-MM-DD HH:MM"
- "end_time": The end time in format "YYYY-MM-DD HH:MM"
- "description": The event description
- "location": The event location (optional)
- "attendees": A list of email addresses for attendees (optional)

JSON Output:
"""

COMMUNICATION_PARSER_PROMPT_TEMPLATE = """
Extract the communication details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "method": Either "call" or "text"
- "recipient": The phone number or contact name
- "message": The message content (for texts) or talking points (for calls)

JSON Output:
"""

SEARCH_QUERY_PARSER_PROMPT_TEMPLATE = """
Extract a clear search query from the following action text:
"{action_text}"

Return just the search query text, nothing else.
"""

WEATHER_PARSER_PROMPT_TEMPLATE = """
Extract the weather request details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "location": The location to get weather for
- "type": Either "current" or "forecast"
- "days": Number of days for forecast (if applicable)

JSON Output:
"""

BOOKING_PARSER_PROMPT_TEMPLATE = """
Extract the booking details from the following action text:
"{action_text}"

Return a JSON object with these fields:
- "service_type": Type of booking (restaurant, hotel, flight, etc.)
- "provider": Name of provider if specified
- "details": All relevant details (date, time, number of people, etc.)
- "preferences": Any user preferences mentioned

JSON Output:
"""

# =============================================================================
# AGENT CLASSES
# =============================================================================

class BaseAgent:
    """Base class for all agents with common functionality."""
    
    def __init__(self, name):
        self.name = name
        logger.info(f"Initialized {name}")
    
    async def run(self, *args, **kwargs):
        """Execute the agent's main functionality."""
        raise NotImplementedError("Subclasses must implement run()")
    
    def _format_simulation_message(self, action_description: str) -> dict:
        """Format a standardized simulation mode message."""
        return {
            "status": "simulated",
            "message": f"[SIMULATION] {action_description}",
            "warning": "This action was simulated due to missing API credentials"
        }
    
    async def validate_credentials(self):
        """Check if all required credentials are available."""
        return True

class KnowledgeAgent(BaseAgent):
    """Agent for retrieving information from internal knowledge base."""
    
    def __init__(self, directory="knowledge_base"):
        super().__init__("KnowledgeAgent")
        self.directory = directory
        self.knowledge = self._load_knowledge()
    
    def _load_knowledge(self):
        """Load all knowledge from text files in the knowledge base directory."""
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            # Create a sample knowledge file
            with open(os.path.join(self.directory, "sample.txt"), "w") as f:
                f.write("This is sample knowledge. Replace with actual information.")
            
        knowledge_data = {}
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    knowledge_data[filename.replace(".txt", "")] = f.read()
                    
        return knowledge_data
    
    async def run(self, query: str) -> dict:
        """Search the knowledge base for information related to the query."""
        logger.info(f"Searching knowledge base for: {query}")
        
        # In a real implementation, this would use a vector database or other search mechanism
        # For now, we'll implement a simple keyword-based search
        results = []
        query_terms = query.lower().split()
        
        for topic, content in self.knowledge.items():
            relevance_score = 0
            for term in query_terms:
                if term in content.lower():
                    relevance_score += content.lower().count(term)
            
            if relevance_score > 0:
                results.append((topic, content, relevance_score))
        
        # Sort by relevance
        results.sort(key=lambda x: x[2], reverse=True)
        
        if not results:
            return {
                "status": "no_results",
                "message": "[NO RESULTS] No relevant information found in the knowledge base",
                "query": query,
                "searched_files": len(self.knowledge)
            }
        
        # Return the most relevant information
        return {
            "status": "success",
            "message": f"[SUCCESS] Found information on '{results[0][0]}'",
            "query": query,
            "best_match": {
                "topic": results[0][0],
                "content_preview": results[0][1][:500] + "..." if len(results[0][1]) > 500 else results[0][1],
                "relevance_score": results[0][2]
            },
            "total_matches": len(results)
        }

class SearchAgent(BaseAgent):
    """Agent for performing web searches."""
    
    def __init__(self):
        super().__init__("SearchAgent")
    
    async def run(self, query: str) -> dict:
        """Perform a web search for the given query."""
        logger.info(f"Searching the web for: {query}")
        
        try:
            # In a real implementation, this would use a search API
            # For demonstration purposes, we'll simulate search results
            await asyncio.sleep(1)
            
            # Simulated search results
            results = [
                {
                    "title": f"Result 1 for {query}",
                    "snippet": f"This is a sample search result for {query}.",
                    "link": "https://example.com/result1"
                },
                {
                    "title": f"Result 2 for {query}",
                    "snippet": f"Another sample result relevant to {query}.",
                    "link": "https://example.com/result2"
                },
                {
                    "title": f"Result 3 for {query}",
                    "snippet": f"A third sample result for the query {query}.",
                    "link": "https://example.com/result3"
                }
            ]
            
            return {
                "status": "simulated",
                "message": f"[SIMULATION] SIMULATION MODE: Web search results for '{query}' are simulated",
                "warning": "Real web search requires a search API integration",
                "query": query,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "status": "error",
                "message": f"[ERROR] Error performing search: {str(e)}",
                "query": query
            }

class SlackAgent(BaseAgent):
    """Agent for posting messages to Slack channels."""
    
    def __init__(self):
        super().__init__("SlackAgent")
        self.api_token = SLACK_BOT_TOKEN
    
    async def validate_credentials(self):
        return bool(self.api_token and not self.api_token.startswith('xoxb-your-slack'))
    
    async def run(self, channel: str, message: str) -> dict:
        """Post a message to a Slack channel."""
        logger.info(f"Posting to Slack channel {channel}: {message}")
        
        if not await self.validate_credentials():
            logger.warning("Slack API token not set. Using simulation mode.")
            await asyncio.sleep(1)
            return self._format_simulation_message(f"Would post '{message}' to Slack channel {channel}")
        
        try:
            # Real Slack API implementation
            import requests
            
            url = "https://slack.com/api/chat.postMessage"
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            payload = {
                "channel": channel,
                "text": message
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            response_data = response.json()
            if response_data.get("ok"):
                return {
                    "status": "success",
                    "message": f"[SUCCESS] Message successfully posted to {channel}",
                    "details": {
                        "channel": channel, 
                        "message": message,
                        "timestamp": response_data.get("ts")
                    }
                }
            else:
                error = response_data.get("error", "Unknown error")
                raise Exception(f"Slack API error: {error}")
            
        except Exception as e:
            logger.error(f"Slack error: {e}")
            raise Exception(f"Error posting to Slack: {str(e)}")

class EmailAgent(BaseAgent):
    """Agent for sending emails via SendGrid."""
    
    def __init__(self):
        super().__init__("EmailAgent")
        self.api_key = SENDGRID_API_KEY
    
    async def validate_credentials(self):
        return bool(self.api_key and not self.api_key.startswith('your_sendgrid'))
    
    async def run(self, to: str, subject: str, body: str) -> dict:
        """Send an email to the specified recipient."""
        logger.info(f"Sending email to {to}: {subject}")
        
        if not await self.validate_credentials():
            logger.warning("SendGrid API key not set. Using simulation mode.")
            await asyncio.sleep(1)
            return self._format_simulation_message(f"Would send email to {to} with subject '{subject}'")
        
        try:
            # Real SendGrid API implementation
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            
            message = Mail(
                from_email='noreply@taskautomation.app',  # You can customize this
                to_emails=to,
                subject=subject,
                html_content=body
            )
            
            sg = SendGridAPIClient(api_key=self.api_key)
            response = sg.send(message)
            
            return {
                "status": "success",
                "message": f"[SUCCESS] Email sent to {to} with subject: {subject}",
                "details": {
                    "to": to, 
                    "subject": subject, 
                    "body_length": len(body),
                    "status_code": response.status_code,
                    "message_id": response.headers.get("X-Message-Id")
                }
            }
            
        except Exception as e:
            logger.error(f"SendGrid error: {e}")
            raise Exception(f"Failed to send email to {to}: {str(e)}")

class CalendarAgent(BaseAgent):
    """Agent for interacting with calendars."""
    
    def __init__(self):
        super().__init__("CalendarAgent")
    
    async def run(self, title: str, start_time: str, end_time: str, 
                 description: str = "", location: str = "", attendees: List[str] = None) -> Dict:
        """Create, update, or delete a calendar event."""
        logger.info(f"Calendar operation: {title} from {start_time} to {end_time}")
        
        # In a real implementation, this would interact with a calendar API
        # For now, we'll simulate the operation
        await asyncio.sleep(1)
        
        event = {
            "status": "simulated",
            "message": f"[SIMULATION] SIMULATION MODE: Calendar event '{title}' would be created",
            "warning": "Calendar operations are simulated - no real calendar API configured",
            "event_details": {
                "title": title,
                "start_time": start_time,
                "end_time": end_time,
                "description": description,
                "location": location,
                "attendees": attendees or [],
                "id": f"event_{hash(title + start_time)}"
            }
        }
        
        return event

class CommunicationAgent(BaseAgent):
    """Agent for making phone calls and sending text messages."""
    
    def __init__(self):
        super().__init__("CommunicationAgent")
        self.account_sid = TWILIO_ACCOUNT_SID
        self.auth_token = TWILIO_AUTH_TOKEN
        self.phone_number = TWILIO_PHONE_NUMBER
    
    async def validate_credentials(self):
        return all([
            self.account_sid and not self.account_sid.startswith('your_twilio'),
            self.auth_token and not self.auth_token.startswith('your_twilio'), 
            self.phone_number and not self.phone_number.startswith('your_twilio')
        ])
    
    async def run(self, method: str, recipient: str, message: str) -> dict:
        """Make a phone call or send a text message."""
        logger.info(f"{method.capitalize()} to {recipient}: {message}")
        
        if not await self.validate_credentials():
            logger.warning("Twilio credentials not set. Using simulation mode.")
            await asyncio.sleep(1)
            return self._format_simulation_message(f"Would {method} {recipient} with message: '{message}'")
            
        try:
            # Real Twilio API implementation
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            if method.lower() == "text":
                # Send SMS using Twilio API
                message_obj = client.messages.create(
                    body=message,
                    from_=self.phone_number,
                    to=recipient
                )
                return {
                    "status": "success",
                    "message": f"[SUCCESS] Text message sent to {recipient}",
                    "details": {
                        "method": "text", 
                        "recipient": recipient, 
                        "message": message,
                        "message_sid": message_obj.sid
                    }
                }
            elif method.lower() == "call":
                # Make voice call using Twilio API
                call = client.calls.create(
                    twiml=f'<Response><Say>{message}</Say></Response>',
                    from_=self.phone_number,
                    to=recipient
                )
                return {
                    "status": "success",
                    "message": f"[SUCCESS] Call initiated to {recipient}",
                    "details": {
                        "method": "call", 
                        "recipient": recipient, 
                        "talking_points": message,
                        "call_sid": call.sid
                    }
                }
            else:
                raise ValueError(f"Unsupported communication method: {method}")
                
        except Exception as e:
            logger.error(f"Communication error: {e}")
            raise Exception(f"Failed to {method} {recipient}: {str(e)}")

class WeatherAgent(BaseAgent):
    """Agent for retrieving weather information via OpenWeatherMap."""
    
    def __init__(self):
        super().__init__("WeatherAgent")
        self.api_key = OPENWEATHERMAP_API_KEY
    
    async def validate_credentials(self):
        return bool(self.api_key and not self.api_key.startswith('your_openweathermap'))
    
    async def run(self, location: str, type: str = "current", days: int = 1) -> Dict:
        """Get current weather or forecast for a location."""
        logger.info(f"Getting {type} weather for {location}")
        
        if not await self.validate_credentials():
            logger.warning("OpenWeatherMap API key not set. Using simulation mode.")
            
            base_data = {
                "status": "simulated",
                "warning": "[SIMULATION] Weather data is simulated due to missing OpenWeatherMap API credentials",
                "location": location
            }
            
            if type.lower() == "current":
                base_data.update({
                    "temperature": 22,
                    "conditions": "Partly Cloudy",
                    "humidity": 65,
                    "wind_speed": 10,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                forecast = []
                for i in range(min(days, 7)):
                    day = datetime.now() + timedelta(days=i)
                    forecast.append({
                        "date": day.strftime("%Y-%m-%d"),
                        "high": 20 + i,
                        "low": 10 + i,
                        "conditions": "Sunny" if i % 2 == 0 else "Cloudy"
                    })
                base_data["forecast"] = forecast
            
            return base_data
        
        try:
            # Real PyOWM (OpenWeatherMap) API implementation
            from pyowm import OWM
            
            owm = OWM(self.api_key)
            mgr = owm.weather_manager()
            
            if type.lower() == "current":
                observation = mgr.weather_at_place(location)
                weather = observation.weather
                
                return {
                    "status": "success",
                    "location": location,
                    "temperature": weather.temperature('celsius')['temp'],
                    "conditions": weather.detailed_status.title(),
                    "humidity": weather.humidity,
                    "wind_speed": weather.wind().get('speed', 0),
                    "pressure": weather.pressure.get('press', 0),
                    "visibility": weather.visibility_distance,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Get forecast
                forecaster = mgr.forecast_at_place(location, '3h')  # 3-hourly forecast
                forecast_data = []
                
                forecast_list = forecaster.forecast.weathers[:min(days * 8, 40)]  # 8 forecasts per day (3h intervals)
                
                for i in range(0, len(forecast_list), 8):  # Group by days
                    if i < len(forecast_list):
                        weather = forecast_list[i]
                        forecast_data.append({
                            "date": weather.reference_time('iso').split('T')[0],
                            "temperature": weather.temperature('celsius')['temp'],
                            "conditions": weather.detailed_status.title(),
                            "humidity": weather.humidity,
                            "wind_speed": weather.wind().get('speed', 0)
                        })
                
                return {
                    "status": "success",
                    "location": location,
                    "forecast": forecast_data
                }
                
        except Exception as e:
            logger.error(f"OpenWeatherMap error: {e}")
            raise Exception(f"Failed to get weather for {location}: {str(e)}")

class BookingAgent(BaseAgent):
    """Agent for booking calendar events via Google Calendar."""
    
    def __init__(self):
        super().__init__("BookingAgent")
        self.credentials_file = GOOGLE_CALENDAR_CREDENTIALS_FILE
        self.token_file = GOOGLE_CALENDAR_TOKEN_FILE
    
    async def validate_credentials(self):
        return bool(
            self.credentials_file and 
            not self.credentials_file.startswith('path/to/') and
            os.path.exists(self.credentials_file)
        )
    
    async def run(self, service_type: str, provider: str = None, 
                 details: Dict = None, preferences: Dict = None) -> Dict:
        """Book an appointment, reservation, or ticket."""
        logger.info(f"Booking {service_type} with {provider}: {details}")
        
        details = details or {}
        preferences = preferences or {}
        
        if not await self.validate_credentials():
            logger.warning("Google Calendar credentials not set. Using simulation mode.")
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            booking_id = f"booking_{hash(str(service_type) + str(provider) + str(datetime.now()))}"
            
            return {
                "status": "simulated",
                "warning": "[SIMULATION] Calendar booking simulated due to missing Google Calendar credentials",
                "booking_id": booking_id,
                "service_type": service_type,
                "provider": provider,
                "booking_status": "would_be_confirmed",
                "details": details,
                "confirmation_code": f"SIM{booking_id[-6:]}",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Real Google Calendar API implementation using gcsa
            from gcsa.google_calendar import GoogleCalendar
            from gcsa.event import Event
            
            # Extract booking details
            title = f"{service_type} with {provider}" if provider else service_type
            start_time = details.get('start_time') or (datetime.now() + timedelta(hours=24))
            end_time = details.get('end_time') or (start_time + timedelta(hours=1))
            description = details.get('description', f"Booking for {service_type}")
            
            # Initialize calendar (you may need to specify calendar ID)
            calendar = GoogleCalendar(credentials_path=self.credentials_file)
            
            # Create calendar event
            event = Event(
                title,
                start=start_time,
                end=end_time,
                description=description,
                location=details.get('location', '')
            )
            
            # Add event to calendar
            created_event = calendar.add_event(event)
            
            return {
                "status": "success",
                "message": f"[SUCCESS] Successfully booked {service_type} with {provider}",
                "booking_id": created_event.event_id,
                "service_type": service_type,
                "provider": provider,
                "booking_status": "confirmed",
                "details": {
                    **details,
                    "calendar_event_id": created_event.event_id,
                    "calendar_link": created_event.html_link if hasattr(created_event, 'html_link') else None
                },
                "confirmation_code": f"CAL{created_event.event_id[-6:]}",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            logger.error(f"Google Calendar booking error: {e}")
            raise Exception(f"Failed to book {service_type} with {provider}: {str(e)}")

# =============================================================================
# TASK ORCHESTRATOR
# =============================================================================

class TaskOrchestrator:
    """Orchestrates the execution of multi-agent tasks based on natural language prompts."""
    
    def __init__(self, task_id: str, prompt: str, ws_manager):
        self.task_id = task_id
        self.prompt = prompt
        self.ws_manager = ws_manager
        self.plan = []
        self.context = {}
        self.start_time = asyncio.get_event_loop().time()
        
        # Initialize agents
        self.agents = {
            "CalendarAgent": CalendarAgent(),
            "CommunicationAgent": CommunicationAgent(),
            "SearchAgent": SearchAgent(),
            "KnowledgeAgent": KnowledgeAgent(),
            "SlackAgent": SlackAgent(),
            "EmailAgent": EmailAgent(),
            "WeatherAgent": WeatherAgent(),
            "BookingAgent": BookingAgent()
        }
        
        # Initialize parsers
        self.parsers = {
            "SlackAgent": SLACK_PARSER_PROMPT_TEMPLATE,
            "EmailAgent": EMAIL_PARSER_PROMPT_TEMPLATE,
            "CalendarAgent": EVENT_PARSER_PROMPT_TEMPLATE,
            "CommunicationAgent": COMMUNICATION_PARSER_PROMPT_TEMPLATE,
            "SearchAgent": SEARCH_QUERY_PARSER_PROMPT_TEMPLATE,
            "WeatherAgent": WEATHER_PARSER_PROMPT_TEMPLATE,
            "BookingAgent": BOOKING_PARSER_PROMPT_TEMPLATE
        }
        
        logger.info(f"Initialized TaskOrchestrator for task {task_id}: {prompt}")

    async def _gemini_request(self, prompt_vars: Dict[str, str], template: str, is_json_output=True):
        """Make a request to the Gemini API with the given prompt template and variables."""
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        
        headers = {"Content-Type": "application/json"}
        final_prompt = template.format(**prompt_vars)
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": final_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 1024,
            }
        }
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                response = requests.post(
                    GEMINI_API_URL, 
                    headers=headers, 
                    json=payload, 
                    timeout=60
                )
                response.raise_for_status()
                response_json = response.json()
                
                content_part = response_json['candidates'][0]['content']['parts'][0]['text']
                
                if is_json_output:
                    # Extract JSON even if it's within markdown code blocks
                    json_text = content_part.strip()
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```")[1].strip()
                    
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from Gemini response: {json_text}")
                        raise ValueError("Invalid JSON response from Gemini API")
                
                return content_part.strip()
                
            except requests.RequestException as e:
                retry_count += 1
                if retry_count == max_retries:
                    logger.error(f"Failed to make Gemini API request after {max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Error processing Gemini API response: {e}")
                raise

    async def execute_plan(self):
        """Create and execute a plan based on the user's prompt."""
        try:
            # Log starting task execution
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Starting task execution: {self.prompt}",
                "log_type": "info",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Generate execution plan
            self.plan = await self._gemini_request(
                {
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": self.prompt
                }, 
                PLANNER_PROMPT_TEMPLATE
            )
            
            # Validate plan
            if not self.plan or not isinstance(self.plan, list):
                raise ValueError("Invalid plan generated. Expected a list of steps.")
            
            # Log the plan
            await self.ws_manager.broadcast(json.dumps({
                "type": "plan", 
                "task_id": self.task_id,
                "plan": self.plan
            }))
            
            # Execute each step
            for index, step in enumerate(self.plan):
                # Format any placeholders in the action with context from previous steps
                try:
                    if "{" in step['action']:
                        step['action'] = step['action'].format(**self.context)
                except KeyError as e:
                    logger.warning(f"Missing context key {e} for step: {step['action']}")
                
                # Execute the step
                step_result = await self._execute_step(step, index)
                
                # Update context with result
                if step_result.get('context_updates'):
                    self.context.update(step_result['context_updates'])
            
            # Log task completion
            execution_time = asyncio.get_event_loop().time() - self.start_time
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Task completed in {execution_time:.2f} seconds",
                "log_type": "success",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Send task completion notification
            await self.ws_manager.broadcast(json.dumps({
                "type": "task_complete",
                "task_id": self.task_id,
                "timestamp": datetime.now().isoformat()
            }))
            
        except Exception as e:
            logger.exception(f"Error executing plan: {e}")
            await self.ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": self.task_id,
                "agent": "System",
                "message": f"Task failed: {str(e)}",
                "log_type": "error",
                "timestamp": datetime.now().isoformat()
            }))

    async def _execute_step(self, step: dict, step_index: int):
        """Execute a single step in the plan."""
        agent_name = step.get('agent', 'UnknownAgent')
        action = step.get('action', 'No action defined')
        description = step.get('description', action)
        
        # Log step starting
        await self.ws_manager.broadcast(json.dumps({
            "type": "status_update",
            "task_id": self.task_id,
            "step_index": step_index,
            "step_action": description,
            "status": "in-progress",
            "timestamp": datetime.now().isoformat()
        }))
        
        await self.ws_manager.broadcast(json.dumps({
            "type": "log",
            "task_id": self.task_id,
            "agent": agent_name,
            "message": f"Starting: {description}",
            "log_type": "info",
            "timestamp": datetime.now().isoformat()
        }))
        
        execution_result = {
            "success": True,
            "message": f"Completed: {description}",
            "context_updates": {}
        }
        
        try:
            # Check if the agent exists
            if agent_name not in self.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            agent = self.agents[agent_name]
            
            # Parse the action according to agent type
            if agent_name in self.parsers:
                # Get the appropriate parser template
                parser_template = self.parsers[agent_name]
                
                # For SearchAgent, we don't need full JSON parsing
                if agent_name == "SearchAgent":
                    query = await self._gemini_request(
                        {"action_text": action}, 
                        parser_template, 
                        is_json_output=False
                    )
                    result = await agent.run(query)
                    
                    # Handle new response format
                    if isinstance(result, dict):
                        status = result.get("status", "unknown")
                        if status == "simulated":
                            execution_result["message"] = result.get("message", f"Search simulated for: '{query}'")
                            execution_result["warning"] = result.get("warning", "")
                        else:
                            execution_result["message"] = f"Found {result.get('result_count', 0)} results for: '{query}'"
                        execution_result["context_updates"]["search_result"] = result
                    else:
                        # Fallback for old format
                        execution_result["message"] = f"Found results for: '{query}'"
                        execution_result["context_updates"]["search_result"] = result
                    
                else:
                    # Parse the action details
                    parsed_details = await self._gemini_request(
                        {"action_text": action},
                        parser_template
                    )
                    
                    # Execute with the parsed details
                    result = await agent.run(**parsed_details)
                    
                    # Handle new response format for all agents
                    if isinstance(result, dict) and "status" in result:
                        status = result.get("status")
                        agent_message = result.get("message", "")
                        
                        if status == "simulated":
                            execution_result["message"] = agent_message or f"{agent_name} operation simulated"
                            execution_result["warning"] = result.get("warning", "")
                            execution_result["simulated"] = True
                        elif status == "success":
                            execution_result["message"] = agent_message or f"{agent_name} operation completed successfully"
                        elif status == "no_results":
                            execution_result["message"] = agent_message or f"{agent_name} found no results"
                        elif status == "error":
                            execution_result["success"] = False
                            execution_result["message"] = agent_message or f"{agent_name} operation failed"
                        else:
                            execution_result["message"] = agent_message or f"{agent_name} operation completed"
                    else:
                        # Fallback for old format
                        execution_result["message"] = f"{agent_name} operation completed"
                    
                    # Update context with result
                    execution_result["context_updates"][f"{agent_name.lower()}_result"] = result
                    
            else:
                # For agents without parsers, just pass the action directly
                result = await agent.run(action)
                
                # Handle new response format
                if isinstance(result, dict) and "status" in result:
                    status = result.get("status")
                    agent_message = result.get("message", "")
                    
                    if status == "simulated":
                        execution_result["message"] = agent_message or f"{agent_name} operation simulated"
                        execution_result["warning"] = result.get("warning", "")
                        execution_result["simulated"] = True
                    elif status == "success":
                        execution_result["message"] = agent_message or f"{agent_name} operation completed successfully"
                    else:
                        execution_result["message"] = agent_message or f"{agent_name} operation completed"
                else:
                    # Fallback for old format
                    execution_result["message"] = f"{agent_name} operation completed"
                
                execution_result["context_updates"][f"{agent_name.lower()}_result"] = result
            
        except Exception as e:
            logger.exception(f"Error executing step {step_index}: {e}")
            execution_result = {
                "success": False,
                "message": f"Failed: {str(e)}",
                "context_updates": {}
            }
        
        # Log step completion with simulation awareness
        status = "completed" if execution_result["success"] else "failed"
        if execution_result.get("simulated"):
            log_type = "warning"
        else:
            log_type = "info" if execution_result["success"] else "error"
        
        await self.ws_manager.broadcast(json.dumps({
            "type": "status_update",
            "task_id": self.task_id,
            "step_index": step_index,
            "step_action": description,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }))
        
        log_message = {
            "type": "log",
            "task_id": self.task_id,
            "agent": agent_name,
            "message": execution_result["message"],
            "log_type": log_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add warning field if this was a simulation
        if execution_result.get("warning"):
            log_message["warning"] = execution_result["warning"]
            
        await self.ws_manager.broadcast(json.dumps(log_message))
        
        return execution_result

# =============================================================================
# HTML TEMPLATE
# =============================================================================


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(task_request: TaskRequest):
    """Create a new task from a natural language prompt."""
    global task_counter
    task_counter += 1
    task_id = f"task_{task_counter}"
    
    logger.info(f"Received task: {task_request.prompt} from user: {task_request.user_id}")
    
    # Create task entry
    tasks_db[task_id] = {
        "prompt": task_request.prompt,
        "user_id": task_request.user_id,
        "status": "scheduled",
        "created_at": asyncio.get_event_loop().time(),
        "updated_at": asyncio.get_event_loop().time(),
    }
    
    # Create and execute orchestrator
    orchestrator = TaskOrchestrator(task_id, task_request.prompt, ws_manager)
    asyncio.create_task(orchestrator.execute_plan())
    
    return TaskResponse(
        task_id=task_id,
        status="scheduled",
        message="Task has been scheduled for execution"
    )

@app.get("/api/tasks/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str):
    """Get details of a specific task."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_db[task_id]

@app.get("/api/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(user_id: Optional[str] = None):
    """List all tasks, optionally filtered by user_id."""
    if user_id:
        return [task for task_id, task in tasks_db.items() if task["user_id"] == user_id]
    return list(tasks_db.values())

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for client connections."""
    await ws_manager.connect(websocket, client_id)
    logger.info(f"WebSocket connection established with client {client_id}")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                logger.info(f"Received message from client {client_id}: {message}")
                
                # Handle different message types here
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong", 
                        "timestamp": json.loads(data).get("timestamp")
                    }))
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
                
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
        logger.info(f"WebSocket connection closed for client {client_id}")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_knowledge_base():
    """Create a sample knowledge base directory if it doesn't exist."""
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        with open(os.path.join("knowledge_base", "sample.txt"), "w") as f:
            f.write("This is sample knowledge. Replace with actual information about your organization, products, or domain.")

if __name__ == "__main__":
    print("="*80)
    print("Starting Multi-Agent Task Automation System")
    print("Open your browser and navigate to http://localhost:8000")
    print("Press CTRL+C to stop the server")
    print("="*80)
    
    # Create knowledge base directory
    create_knowledge_base()
    
    try:
        # Start the FastAPI application
        logger.info("Starting FastAPI server...")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except Exception as e:
        print(f"ERROR: Failed to start server: {e}")
        import traceback
        traceback.print_exc()
