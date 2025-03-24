from mcp.server.fastmcp import FastMCP
import os
import sys
import time
import signal
import smtplib
import requests  
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import imaplib
import email
from email.header import decode_header
import asyncio
import datetime as dt
from threading import Thread


load_dotenv()


SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")


scheduled_emails = []

def signal_handler(sig, frame):
    print("Thanks for using Maitreya's server...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


mcp = FastMCP("gmail-mcp")


mcp.settings.port = 8000


print(f"[INIT].... → SMTP_USERNAME available: {SMTP_USERNAME is not None}")
print(f"[INIT].... → SMTP_PASSWORD available: {SMTP_PASSWORD is not None}")

def send_email(recipient, subject, body, attachment_path=None):
    """Send an email via Gmail SMTP"""
    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USERNAME
        msg["To"] = recipient
        msg["Subject"] = subject

     
        msg.attach(MIMEText(body, "plain"))

       
        if attachment_path:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)

       
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.sendmail(SMTP_USERNAME, recipient, msg.as_string())
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"

def download_attachment_from_url(attachment_url, attachment_filename):
    temp_dir = "temp_attachments"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, attachment_filename)
    response = requests.get(attachment_url)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path

def get_pre_staged_attachment(attachment_name):
    attachment_dir = "available_attachments"
    file_path = os.path.join(attachment_dir, attachment_name)
    if os.path.exists(file_path):
        return file_path
    else:
        return None

@mcp.tool()
def send_email_tool(recipient, subject, body, 
                    attachment_path=None, 
                    attachment_url=None, 
                    attachment_name=None):
    """
    Send an email via Gmail SMTP.
    
    Parameters:
    - recipient: The email address to send the email to.
    - subject: The email subject.
    - body: The email body text.
    - attachment_path: Optional direct file path for an attachment.
    - attachment_url: Optional URL from which to download an attachment.
    - attachment_name: Optional filename for the attachment.
    
    Priority:
      1. If attachment_url is provided (and attachment_name for filename), download the file.
      2. Else if attachment_name is provided, try to load it from the 'available_attachments' directory.
      3. Otherwise, use attachment_path if provided.
    """
    final_attachment_path = attachment_path
    # Use URL-based attachment if provided
    if attachment_url and attachment_name:
        try:
            final_attachment_path = download_attachment_from_url(attachment_url, attachment_name)
        except Exception as e:
            return f"Failed to download attachment from URL: {e}"
    # Otherwise, use pre-staged attachment if specified
    elif attachment_name:
        final_attachment_path = get_pre_staged_attachment(attachment_name)
        if not final_attachment_path:
            return f"Error: Attachment '{attachment_name}' not found in pre-staged directory."
    
    return send_email(recipient, subject, body, final_attachment_path)

@mcp.tool()
def schedule_email(recipient, subject, body, 
                    attachment_path=None, 
                    attachment_url=None, 
                    attachment_name=None,
                    schedule_time=None):
    """
    Schedule an email to be sent at a specified time.
    
    Parameters:
    - recipient: The email address to send the email to.
    - subject: The email subject.
    - body: The email body text.
    - attachment_path: Optional direct file path for an attachment.
    - attachment_url: Optional URL from which to download an attachment.
    - attachment_name: Optional filename for the attachment.
    - schedule_time: The time at which to send the email (in ISO 8601 format).
    """
    if not schedule_time:
        return "Error: Schedule time is required."
    
    schedule_time = dt.datetime.fromisoformat(schedule_time)
    if schedule_time < dt.datetime.now():
        return "Error: Schedule time is in the past."
    
    scheduled_emails.append({
        "recipient": recipient,
        "subject": subject,
        "body": body,
        "attachment_path": attachment_path,
        "attachment_url": attachment_url,
        "attachment_name": attachment_name,
        "schedule_time": schedule_time
    })
    
    return "Email scheduled successfully."

async def send_scheduled_emails():
    while True:
        current_time = dt.datetime.now()
        emails_to_send = [email for email in scheduled_emails if email["schedule_time"] <= current_time]
        for email in emails_to_send:
            final_attachment_path = email["attachment_path"]
            
            if email["attachment_url"] and email["attachment_name"]:
                try:
                    final_attachment_path = download_attachment_from_url(email["attachment_url"], email["attachment_name"])
                except Exception as e:
                    print(f"Failed to download attachment from URL: {e}")
            
            elif email["attachment_name"]:
                final_attachment_path = get_pre_staged_attachment(email["attachment_name"])
                if not final_attachment_path:
                    print(f"Error: Attachment '{email['attachment_name']}' not found in pre-staged directory.")
            
            send_email(email["recipient"], email["subject"], email["body"], final_attachment_path)
            scheduled_emails.remove(email)
        
        await asyncio.sleep(60)  

@mcp.tool()
def start_scheduled_email_service():
    """
    Start the scheduled email service.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(send_scheduled_emails())
    loop.run_forever()

@mcp.tool()
def fetch_recent_emails(folder="INBOX", limit=10):
    """
    Fetch the most recent emails from a specified folder.
    
    Parameters:
    - folder: The email folder to fetch from (default: "INBOX")
    - limit: Maximum number of emails to fetch (default: 10)
    """
    try:
        # Connect to Gmail IMAP
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(SMTP_USERNAME, SMTP_PASSWORD)
        mail.select(folder)
        
        
        result, data = mail.search(None, "ALL")
        if not data or not data[0]:
            return "No emails found in the specified folder."
            
        email_ids = data[0].split()
        latest_email_ids = email_ids[-limit:] if len(email_ids) > limit else email_ids
        
        emails = []
        for email_id in reversed(latest_email_ids):
            result, data = mail.fetch(email_id, "(RFC822)")
            raw_email = data[0][1]
            
           
            msg = email.message_from_bytes(raw_email)
            
            
            subject = decode_header(msg["Subject"])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
                
            
            from_ = msg.get("From", "")
            
           
            date = msg.get("Date", "")
            
            emails.append({
                "id": email_id.decode(),
                "from": from_,
                "subject": subject,
                "date": date
            })
        
        mail.close()
        mail.logout()
        
        if not emails:
            return "No emails found in the specified folder."
        
        result = "Recent emails:\n\n"
        for i, email_data in enumerate(emails, 1):
            result += f"{i}. From: {email_data['from']}\n"
            result += f"   Subject: {email_data['subject']}\n"
            result += f"   Date: {email_data['date']}\n"
            result += f"   ID: {email_data['id']}\n\n"
            
        return result
    except Exception as e:
        return f"Failed to fetch emails: {e}"

if __name__ == "__main__":
    try:
        print("Starting MCP server 'gmail-mcp' on 127.0.0.1:8000")
        
        # Start the scheduled email service in a background thread
        scheduler_thread = Thread(target=start_scheduled_email_service)
        scheduler_thread.daemon = True  # This makes the thread exit when the main program exits
        scheduler_thread.start()
        print("Email scheduler service started in background")
        
        # Run the server using SSE transport
        mcp.run(transport="sse")
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
