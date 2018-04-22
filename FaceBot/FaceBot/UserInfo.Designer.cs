namespace FaceBot
{
    partial class UserInfo
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
            this.UserNameLabel = new System.Windows.Forms.Label();
            this.URLLabel = new System.Windows.Forms.Label();
            this.UserNameTextBox = new System.Windows.Forms.TextBox();
            this.URLTextBox = new System.Windows.Forms.TextBox();
            this.SaveButton = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // UserNameLabel
            // 
            this.UserNameLabel.AutoSize = true;
            this.UserNameLabel.Location = new System.Drawing.Point(28, 36);
            this.UserNameLabel.Name = "UserNameLabel";
            this.UserNameLabel.Size = new System.Drawing.Size(57, 13);
            this.UserNameLabel.TabIndex = 0;
            this.UserNameLabel.Text = "UserName";
            // 
            // URLLabel
            // 
            this.URLLabel.AutoSize = true;
            this.URLLabel.Location = new System.Drawing.Point(28, 89);
            this.URLLabel.Name = "URLLabel";
            this.URLLabel.Size = new System.Drawing.Size(29, 13);
            this.URLLabel.TabIndex = 1;
            this.URLLabel.Text = "URL";
            // 
            // UserNameTextBox
            // 
            this.UserNameTextBox.Location = new System.Drawing.Point(120, 36);
            this.UserNameTextBox.Name = "UserNameTextBox";
            this.UserNameTextBox.Size = new System.Drawing.Size(230, 20);
            this.UserNameTextBox.TabIndex = 2;
            // 
            // URLTextBox
            // 
            this.URLTextBox.Location = new System.Drawing.Point(120, 89);
            this.URLTextBox.Name = "URLTextBox";
            this.URLTextBox.Size = new System.Drawing.Size(230, 20);
            this.URLTextBox.TabIndex = 3;
            // 
            // SaveButton
            // 
            this.SaveButton.Location = new System.Drawing.Point(149, 185);
            this.SaveButton.Name = "SaveButton";
            this.SaveButton.Size = new System.Drawing.Size(75, 23);
            this.SaveButton.TabIndex = 4;
            this.SaveButton.Text = "Save";
            this.SaveButton.UseVisualStyleBackColor = true;
            this.SaveButton.Click += new System.EventHandler(this.SaveButton_Click);
            // 
            // UserInfo
            // 
            this.ClientSize = new System.Drawing.Size(376, 261);
            this.Controls.Add(this.SaveButton);
            this.Controls.Add(this.URLTextBox);
            this.Controls.Add(this.UserNameTextBox);
            this.Controls.Add(this.URLLabel);
            this.Controls.Add(this.UserNameLabel);
            this.Name = "UserInfo";
            this.Text = "User Info";
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label UserNameLabel;
        private System.Windows.Forms.Label URLLabel;
        private System.Windows.Forms.TextBox UserNameTextBox;
        private System.Windows.Forms.TextBox URLTextBox;
        private System.Windows.Forms.Button SaveButton;
    }
}