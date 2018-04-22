namespace FaceBot
{
    partial class FaceBot
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.VideoFrameImageBox = new Emgu.CV.UI.ImageBox();
            this.AddUserButton = new System.Windows.Forms.Button();
            this.FPSTextBox = new System.Windows.Forms.TextBox();
            this.FPSLabel = new System.Windows.Forms.Label();
            ((System.ComponentModel.ISupportInitialize)(this.VideoFrameImageBox)).BeginInit();
            this.SuspendLayout();
            // 
            // VideoFrameImageBox
            // 
            this.VideoFrameImageBox.Location = new System.Drawing.Point(12, 12);
            this.VideoFrameImageBox.Name = "VideoFrameImageBox";
            this.VideoFrameImageBox.Size = new System.Drawing.Size(1500, 800);
            this.VideoFrameImageBox.TabIndex = 2;
            this.VideoFrameImageBox.TabStop = false;
            // 
            // AddUserButton
            // 
            this.AddUserButton.Location = new System.Drawing.Point(1322, 38);
            this.AddUserButton.Name = "AddUserButton";
            this.AddUserButton.Size = new System.Drawing.Size(75, 23);
            this.AddUserButton.TabIndex = 3;
            this.AddUserButton.Text = "Add User";
            this.AddUserButton.UseVisualStyleBackColor = true;
            this.AddUserButton.Click += new System.EventHandler(this.AddUserButton_Click);
            // 
            // FPSTextBox
            // 
            this.FPSTextBox.Location = new System.Drawing.Point(1341, 703);
            this.FPSTextBox.Name = "FPSTextBox";
            this.FPSTextBox.Size = new System.Drawing.Size(30, 20);
            this.FPSTextBox.TabIndex = 4;
            // 
            // FPSLabel
            // 
            this.FPSLabel.AutoSize = true;
            this.FPSLabel.Location = new System.Drawing.Point(1342, 687);
            this.FPSLabel.Name = "FPSLabel";
            this.FPSLabel.Size = new System.Drawing.Size(27, 13);
            this.FPSLabel.TabIndex = 7;
            this.FPSLabel.Text = "FPS";
            // 
            // FaceBotForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1432, 741);
            this.Controls.Add(this.FPSLabel);
            this.Controls.Add(this.FPSTextBox);
            this.Controls.Add(this.AddUserButton);
            this.Controls.Add(this.VideoFrameImageBox);
            this.Name = "FaceBotForm";
            this.Text = "FaceBot";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.FaceBotForm_FormClosing);
            ((System.ComponentModel.ISupportInitialize)(this.VideoFrameImageBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button AddUserButton;
        private System.Windows.Forms.TextBox FPSTextBox;
        private System.Windows.Forms.Label FPSLabel;
        private Emgu.CV.UI.ImageBox VideoFrameImageBox;
    }
}

