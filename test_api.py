#!/usr/bin/env python3
import requests
import json

# Test data
test_resume = """
John Doe
Software Engineer

SUMMARY
Experienced software engineer with 5+ years developing web applications using JavaScript, React, and Node.js. Proven track record of delivering scalable solutions and leading cross-functional teams.

WORK EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2021-2024
• Led development of customer-facing web application serving 100K+ users
• Implemented React components and Redux state management
• Collaborated with product managers and designers to deliver features
• Mentored 2 junior developers and conducted code reviews

Software Engineer | StartupXYZ | 2019-2021
• Built RESTful APIs using Node.js and Express framework
• Integrated third-party payment systems (Stripe, PayPal)
• Optimized database queries reducing response times by 40%
• Participated in agile development process and sprint planning

EDUCATION
Bachelor of Science in Computer Science | UC Berkeley | 2019

SKILLS
Languages: JavaScript, TypeScript, Python, Java
Frontend: React, Vue.js, HTML5, CSS3, Sass
Backend: Node.js, Express, Django, Spring Boot
Databases: PostgreSQL, MongoDB, Redis
Tools: Git, Docker, AWS, Jenkins, Jira
"""

test_job_description = """
Senior Full Stack Developer
InnovateTech Solutions

We are seeking a Senior Full Stack Developer to join our growing team and help build our cloud-based SaaS platform that serves enterprise customers.

Responsibilities:
• Design and develop scalable web applications using modern JavaScript frameworks
• Build and maintain RESTful APIs and microservices
• Collaborate with product team to translate requirements into technical solutions
• Optimize application performance and ensure high availability
• Mentor junior developers and participate in code reviews
• Work with DevOps team on deployment and monitoring

Required Qualifications:
• 4+ years of experience in full stack development
• Strong proficiency in JavaScript/TypeScript and React
• Experience with Node.js and Express framework
• Knowledge of SQL and NoSQL databases
• Experience with cloud platforms (AWS, Azure, or GCP)
• Understanding of Agile development methodologies
• Bachelor's degree in Computer Science or related field

Preferred Qualifications:
• Experience with Docker and Kubernetes
• Knowledge of GraphQL
• Previous experience in SaaS or enterprise software
• Experience with automated testing frameworks
• Strong communication and leadership skills
"""

def test_analyze_endpoint():
    url = "http://localhost:8080/analyze"
    
    payload = {
        "resume_text": test_resume,
        "job_description": test_job_description,
        "company_name": "InnovateTech Solutions"
    }
    
    try:
        print("Testing /analyze endpoint...")
        response = requests.post(url, json=payload, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(f"Analysis Score: {result['analysis']['score']}")
            print(f"Score Rationale: {result['analysis']['scoreRationale']}")
            print(f"Strengths: {len(result['analysis']['strengths'])} items")
            print(f"Improvements: {len(result['analysis']['improvements'])} items")
            print("\nFirst improvement suggestion:")
            if result['analysis']['improvements']:
                imp = result['analysis']['improvements'][0]
                print(f"  - {imp['suggestion']}: {imp['explanation']}")
        else:
            print(f"❌ Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_analyze_endpoint()