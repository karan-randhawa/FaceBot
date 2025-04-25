# 🧠 FaceBot

A lightweight and secure, **ML-equipped**, user authentication system using **face recognition**. Built with C# and OpenCV, FaceBot provides an intuitive way to verify user identity through camera-based facial scans.

## 🔍 Features

- 📸 Face capture and recognition  
- 🔐 Secure authentication flow  
- 🗃️ User registration with face data  
- 🧠 OpenCV-powered detection  
- 🛠️ Extensible for enterprise or consumer use cases  

## 🧪 Tech Stack

- **Language**: C#  
- **Libraries**: OpenCV (via Emgu CV wrapper)  
- **UI**: Windows Forms  
- **Database**: Local file-based storage (can be extended to SQL)
- **ML Dataset Training**: Haar Cascades

## 🚀 Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/karan-randhawa/FaceBot.git
2. Open the solution in Visual Studio.
3. Restore NuGet packages.
4. Run the project and follow on-screen instructions to register or authenticate via webcam.
   
## ⚠️ Make sure your device has camera access enabled.

## Repo Structure
FaceBot\
│\
├── FaceBot            # Main application logic\
├── Database           # Stores face images and data\
└── README.md

## 📈 Future Roadmap
 - Migrate to cloud-based facial models
 - Add liveness detection (anti-spoofing)
 - Multi-user dashboard and admin access
 - API for integration with third-party systems

### Made with ❤️ by [Karan Randhawa](https://www.github.com/karan-randhawa)
