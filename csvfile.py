import csv
import random

# Define eight job roles with realistic details
job_roles = {
    "Cloud Engineer": {
        "skills": ["Kubernetes", "Terraform", "Cloud Security", "Azure", "AWS", "CI/CD", "Docker", "Monitoring", "Serverless"],
        "exp_range": (3, 10),
        "keywords": ["DevOps", "Cloud Migration", "Infrastructure as Code", "Scalability", "Automation"]
    },
    "Software Engineer": {
        "skills": ["Java", "C++", "Python", "Microservices", "Docker", "Git", "REST APIs", "Spring Boot", "Agile"],
        "exp_range": (1, 8),
        "keywords": ["Scalability", "Code Quality", "Design Patterns", "Unit Testing", "Agile Development"]
    },
    "Cybersecurity Analyst": {
        "skills": ["Network Security", "Cryptography", "Penetration Testing", "SOC", "Incident Response", "Risk Assessment", "Vulnerability Management"],
        "exp_range": (2, 12),
        "keywords": ["Threat Intelligence", "Compliance", "Forensics", "SIEM", "Mitigation"]
    },
    "Data Scientist": {
        "skills": ["Python", "R", "Machine Learning", "Deep Learning", "Statistics", "Data Visualization", "SQL", "TensorFlow", "PyTorch"],
        "exp_range": (1, 10),
        "keywords": ["Predictive Modeling", "Big Data", "ETL", "Data Mining", "A/B Testing"]
    },
    "DevOps Engineer": {
        "skills": ["Jenkins", "Docker", "Kubernetes", "AWS", "Terraform", "Ansible", "CI/CD", "Monitoring", "Linux"],
        "exp_range": (2, 10),
        "keywords": ["Automation", "Infrastructure as Code", "Cloud", "Agile", "Scalability"]
    },
    "Network Engineer": {
        "skills": ["Routing", "Switching", "VPN", "Cisco", "Network Security", "Firewalls", "LAN", "WAN", "SDN"],
        "exp_range": (2, 12),
        "keywords": ["Network Design", "Troubleshooting", "Capacity Planning", "QoS", "Monitoring"]
    },
    "Machine Learning Engineer": {
        "skills": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "Data Preprocessing", "Scikit-Learn", "NLP", "Computer Vision", "Model Deployment"],
        "exp_range": (2, 10),
        "keywords": ["Algorithm Development", "Model Tuning", "Big Data", "AI", "Research"]
    },
    "IT Support Specialist": {
        "skills": ["Windows", "Linux", "Troubleshooting", "Customer Service", "Active Directory", "Network Support", "Help Desk", "Remote Support", "Software Installation"],
        "exp_range": (1, 8),
        "keywords": ["Technical Support", "Incident Management", "Ticketing", "Hardware", "User Training"]
    }
}

# Define output file name and total number of entries
output_file = "job_requirements_generated.csv"
total_entries = 4000

fieldnames = ["Job Role", "Required Skills", "Experience", "Keywords"]

def generate_entry():
    # Randomly pick one of the job roles
    role = random.choice(list(job_roles.keys()))
    details = job_roles[role]

    # Randomly sample skills (at least 50% of the skills, but with random order)
    num_skills = random.randint(max(1, len(details["skills"]) // 2), len(details["skills"]))
    skills_sample = random.sample(details["skills"], num_skills)
    random.shuffle(skills_sample)
    required_skills = ", ".join(skills_sample)

    # Generate experience string (e.g., "5+ years")
    exp_years = random.randint(details["exp_range"][0], details["exp_range"][1])
    experience = f"{exp_years}+ years"

    # Randomly sample keywords (choose 2 to all available)
    num_keywords = random.randint(2, len(details["keywords"]))
    keywords_sample = random.sample(details["keywords"], num_keywords)
    random.shuffle(keywords_sample)
    keywords = ", ".join(keywords_sample)

    return {
        "Job Role": role,
        "Required Skills": required_skills,
        "Experience": experience,
        "Keywords": keywords
    }

with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    seen_entries = set()  # To help avoid repeats
    while len(seen_entries) < total_entries:
        entry = generate_entry()
        # Create a unique key from the entry's values
        key = (entry["Job Role"], entry["Required Skills"], entry["Experience"], entry["Keywords"])
        if key not in seen_entries:
            seen_entries.add(key)
            writer.writerow(entry)

print(f"Generated {output_file} with {total_entries} unique entries.")
