"""
Generates a comprehensive, professional PDF manual for setting up and running
the RL-Based Smart Traffic Light Controller project from GitHub on another machine.
"""
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

PDF_OUTPUT_PATH = os.path.join(os.getcwd(), "Project_Execution_Guide.pdf")

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, total_pages):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "RL Smart Traffic Light Controller — Setup & Execution Guide")
            self.drawRightString(612 - 54, 750, "github.com/SauravLamba17/RL-Smart-Traffic-Light")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Confidential & Academic Reference · Reinforcement Learning Research")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf():
    # 54pt = 0.75 in margins
    doc = SimpleDocTemplate(
        PDF_OUTPUT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()

    # Custom Palette
    PRIMARY = colors.HexColor("#0f172a")   # Dark Navy
    SECONDARY = colors.HexColor("#0284c7") # Vibrant Sky/Teal
    MUTED = colors.HexColor("#475569")     # Slate Gray
    BG_CODE = colors.HexColor("#f8fafc")   # Very light blue-gray
    BORDER_CODE = colors.HexColor("#cbd5e1")

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        alignment=TA_LEFT,
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        alignment=TA_LEFT,
        spaceAfter=8
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=PRIMARY,
        spaceAfter=4,
        alignment=TA_JUSTIFY
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        fontName='Helvetica',
        fontSize=8,
        leading=11.5,
        textColor=PRIMARY,
        leftIndent=12,
        spaceAfter=2.5
    )

    code_style = ParagraphStyle(
        'CodeBlock',
        fontName='Courier-Bold',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#0f172a"),
        leftIndent=4
    )

    note_style = ParagraphStyle(
        'NoteText',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e3a8a"),
        spaceAfter=3
    )

    story = []

    # ==================== PAGE 1 ====================
    # Title Block
    story.append(Paragraph("Smart Traffic Light Controller using RL", title_style))
    story.append(Paragraph("Step-by-Step Installation & Execution Manual for Any Computer", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=SECONDARY, spaceAfter=8))

    # Meta Info Card
    meta_data = [
        [
            Paragraph("<b>Repository:</b> github.com/SauravLamba17/RL-Smart-Traffic-Light", body_style),
            Paragraph("<b>Author:</b> Saurav Lamba", body_style),
        ],
        [
            Paragraph("<b>Environment:</b> Python 3.10 – 3.14 (Cross-Platform)", body_style),
            Paragraph("<b>Core Stack:</b> Q-Learning, Pygame-CE, Streamlit", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[260, 244])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#e2e8f0")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 6))

    # Section 1: Overview
    story.append(Paragraph("1. Project Overview & Capabilities", h1_style))
    story.append(Paragraph(
        "This project replaces rigid fixed-timer intersection lights with an autonomous "
        "<b>Tabular Q-Learning</b> reinforcement learning agent. The agent observes 4-lane queue sizes, "
        "detects emergency vehicle presence (Tier 1 Fire Trucks & Tier 2 Ambulances), and optimizes signal timing "
        "in real time. The repository comes fully bundled with a pre-trained model, animated Pygame desktop simulation, "
        "and a rich 5-page Streamlit analytical web dashboard.",
        body_style
    ))

    # Section 2: Prerequisites
    story.append(Paragraph("2. System Prerequisites", h1_style))
    story.append(Paragraph("Ensure the target machine meets the following basic requirements:", body_style))
    story.append(Paragraph("• <b>Operating System:</b> Windows 10/11, macOS (Intel or Apple Silicon), or Linux (Ubuntu/Debian/Fedora).", bullet_style))
    story.append(Paragraph("• <b>Python:</b> Python 3.10 to Python 3.14. <i>(Ensure 'Add python.exe to PATH' was checked during setup).</i>", bullet_style))
    story.append(Paragraph("• <b>Hardware:</b> Any dual-core CPU with at least 4 GB RAM. <b>No GPU required</b> (&lt; 0.1 ms lookup per frame).", bullet_style))
    story.append(Paragraph("• <b>Git:</b> Installed for cloning (or simply use direct ZIP download).", bullet_style))
    story.append(Spacer(1, 4))

    # Section 3: Step-by-Step Setup
    story.append(Paragraph("3. Step-by-Step Setup Instructions", h1_style))
    
    # Step 3.1
    story.append(Paragraph("Step 3.1: Download or Clone the Repository", h2_style))
    story.append(Paragraph("Open a Terminal (macOS/Linux) or PowerShell / Command Prompt (Windows) and run:", body_style))
    
    cmd1 = Table([[Paragraph("git clone https://github.com/SauravLamba17/RL-Smart-Traffic-Light.git<br/>cd RL-Smart-Traffic-Light", code_style)]], colWidths=[504])
    cmd1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CODE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_CODE),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(cmd1)
    story.append(Paragraph("<i>Alternative without Git:</i> Visit GitHub &rarr; green <b>Code</b> button &rarr; <b>Download ZIP</b>, and extract.", note_style))

    # Step 3.2
    story.append(Paragraph("Step 3.2: Create and Activate a Virtual Environment (Recommended)", h2_style))
    story.append(Paragraph("Creating a virtual environment ensures zero library conflicts with existing system packages:", body_style))

    cmd2_content = """<b>Windows (PowerShell or Command Prompt):</b><br/>
python -m venv venv<br/>
.\\venv\\Scripts\\activate<br/>
<b>macOS / Linux (Bash or Zsh):</b><br/>
python3 -m venv venv<br/>
source venv/bin/activate"""
    cmd2 = Table([[Paragraph(cmd2_content, code_style)]], colWidths=[504])
    cmd2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CODE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_CODE),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(cmd2)

    # Step 3.3
    story.append(Paragraph("Step 3.3: Install Dependencies", h2_style))
    story.append(Paragraph("Install all runtime dependencies with a single pip command:", body_style))
    
    cmd3 = Table([[Paragraph("pip install -r requirements.txt", code_style)]], colWidths=[504])
    cmd3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CODE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_CODE),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(cmd3)
    story.append(Paragraph("<i>Packages:</i> <code>pygame-ce</code>, <code>streamlit</code>, <code>pandas</code>, <code>plotly</code>, <code>matplotlib</code>, <code>numpy</code>.", note_style))

    # Clean break to Page 2
    story.append(PageBreak())

    # ==================== PAGE 2 ====================
    # Section 4: Running the System
    story.append(Paragraph("4. How to Run the Software", h1_style))
    story.append(Paragraph(
        "The project is engineered for zero-friction launch. You can run the entire system with one single command, "
        "or launch individual components independently.",
        body_style
    ))

    story.append(Paragraph("Method A: The Unified Launcher (Recommended)", h2_style))
    story.append(Paragraph("Run the all-in-one execution controller:", body_style))
    
    cmd4 = Table([[Paragraph("python run.py", code_style)]], colWidths=[504])
    cmd4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CODE),
        ('BOX', (0,0), (-1,-1), 1, BORDER_CODE),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(cmd4)
    story.append(Paragraph("<i>On Windows, you can also simply double-click <b>START_SIMULATION.bat</b>.</i>", note_style))

    story.append(Paragraph("<b>What happens automatically:</b>", body_style))
    story.append(Paragraph("1. <b>Model Verification:</b> Checks <code>models/trained_q_table.pkl</code> (bundled in repo, ready out-of-the-box).", bullet_style))
    story.append(Paragraph("2. <b>Web Dashboard:</b> Starts Streamlit on port 8501 and opens browser to <code>http://localhost:8501</code>.", bullet_style))
    story.append(Paragraph("3. <b>Interactive Desktop Simulation:</b> Launches the animated Pygame launch screen. Use &uarr;/&darr; and press <b>ENTER</b>.", bullet_style))
    story.append(Paragraph("4. <b>Clean Shutdown:</b> Closing the Pygame window automatically halts all background processes.", bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("Method B: Launching Standalone Components", h2_style))
    
    comp_data = [
        ["Target Action", "Command Line Execution", "Description"],
        ["Live Simulation", "python main.py --simulate", "Directly opens animated Pygame window."],
        ["Web Analytics", "streamlit run streamlit_app/app.py", "Opens the 5-page analytical dashboard."],
        ["Retrain Agent", "python main.py --train", "Trains model from scratch (2500 ep, ~2.5 min)."],
        ["Benchmark Eval", "python main.py --eval", "Runs 50-episode headless statistical benchmark."],
        ["Generate Plots", "python main.py --plot", "Exports high-res PNG plots into analytics/plots/."]
    ]
    comp_table = Table([[Paragraph(f"<b>{c}</b>" if i==0 else c, body_style) for c in row] for i, row in enumerate(comp_data)], colWidths=[90, 190, 224])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(comp_table)
    story.append(Spacer(1, 6))

    # Section 5: Keyboard Controls
    story.append(Paragraph("5. Interactive Pygame Simulation Controls", h1_style))
    story.append(Paragraph("While the live side-by-side simulation is active, interact using these keys:", body_style))

    key_data = [
        ["Key Binding", "Functionality & Demonstration Highlight"],
        ["1, 2, 3, 4", "Dispatches an <b>Ambulance (Tier 2 Priority)</b> into North, South, East, or West lane."],
        ["5, 6, 7, 8", "Dispatches a <b>Fire Truck (Tier 1 Critical)</b> into North, South, East, or West lane."],
        ["F  or  C", "<b>Emergency Conflict Test:</b> Simultaneously spawns Fire Truck (North) and Ambulance (East) to demonstrate Tier 1 vs Tier 2 priority arbitration!"],
        ["SPACE", "Pauses / Resumes the simulation physics."],
        ["UP / DOWN", "Cycles simulation speed: <b>1x &rarr; 2x &rarr; 4x &rarr; 8x</b>."],
        ["T", "Cycles traffic patterns (Balanced, Rush Hour, Heavy N-S, Heavy E-W, Light)."],
        ["R", "Resets simulation with a fresh synchronized random seed."],
        ["ESC / Q", "Exits the simulation cleanly."]
    ]
    key_table = Table([[Paragraph(f"<b>{r[0]}</b>", code_style if i>0 else body_style), Paragraph(f"<b>{r[1]}</b>" if i==0 else r[1], body_style)] for i, r in enumerate(key_data)], colWidths=[80, 424])
    key_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(key_table)

    # Clean break to Page 3
    story.append(PageBreak())

    # ==================== PAGE 3 ====================
    # Section 6: Streamlit Guide
    story.append(Paragraph("6. Streamlit Analytics Dashboard Structure", h1_style))
    story.append(Paragraph("Navigate the browser interface at <code>http://localhost:8501</code> to explore each dedicated section:", body_style))
    story.append(Paragraph("• <b>1_Overview:</b> Problem statement, Markov Decision Process (MDP) state-action formalization, and reward architecture.", bullet_style))
    story.append(Paragraph("• <b>2_Training:</b> Interactive Plotly charts of episode reward convergence, rolling average wait time, and epsilon decay curves.", bullet_style))
    story.append(Paragraph("• <b>3_Benchmark:</b> Head-to-head empirical comparison (Wait time: -28%, Throughput: +4.1%, Emergency: -81.4%).", bullet_style))
    story.append(Paragraph("• <b>4_Emergency:</b> Preemption logic explanation and conflict resolution workflow.", bullet_style))
    story.append(Paragraph("• <b>5_Dataset:</b> Kaggle Urban Traffic Light Control dataset calibration proving simulation fidelity against real-world traffic data.", bullet_style))
    story.append(Spacer(1, 8))

    # Section 7: Troubleshooting
    story.append(Paragraph("7. Frequently Encountered Issues & Troubleshooting", h1_style))

    faq_data = [
        [
            Paragraph("<b>Issue Observed</b>", body_style),
            Paragraph("<b>Root Cause & Instant Resolution</b>", body_style)
        ],
        [
            Paragraph("<code>'python' is not recognized</code>", code_style),
            Paragraph("Python was not added to system PATH during install. Re-run installer, select 'Modify', check 'Add Python to environment variables', and restart terminal.", body_style)
        ],
        [
            Paragraph("<code>Port 8501 is in use</code>", code_style),
            Paragraph("Another Streamlit instance is running. Kill existing process or launch on alternate port: <code>streamlit run streamlit_app/app.py --server.port 8502</code>.", body_style)
        ],
        [
            Paragraph("<code>Pygame DLL error (App Control)</code>", code_style),
            Paragraph("Certain Windows Defender Application Control policies block standard Pygame DLLs. Our <code>requirements.txt</code> uses <b>pygame-ce</b> which resolves this.", body_style)
        ],
        [
            Paragraph("<code>Permission denied (PowerShell script)</code>", code_style),
            Paragraph("PowerShell restricts unsigned scripts by default. Run: <code>Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass</code> before activating the venv.", body_style)
        ],
        [
            Paragraph("<code>ModuleNotFoundError: No module...</code>", code_style),
            Paragraph("Virtual environment was not activated or dependencies were not installed. Ensure virtual environment is active (<code>(venv)</code> prefix in prompt) and run <code>pip install -r requirements.txt</code>.", body_style)
        ]
    ]
    faq_table = Table(faq_data, colWidths=[140, 364])
    faq_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(faq_table)
    story.append(Spacer(1, 14))

    # Concluding signature
    story.append(Paragraph("<b>End of Setup Manual.</b> For issues, code updates, or star support, visit: <u>https://github.com/SauravLamba17/RL-Smart-Traffic-Light</u>", note_style))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[OK] Successfully built polished PDF at: {PDF_OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
