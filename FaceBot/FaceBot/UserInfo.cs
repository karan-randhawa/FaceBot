using System;
using System.Windows.Forms;

namespace FaceBot
{
    public partial class UserInfo : Form
    {
        public UserInfo()
        {
            InitializeComponent();
        }

        // Called when a user has been recognized when trying to add another face, to autofill the UI fields with their values.
        public void Populate(string username, string url)
        {
            UserNameTextBox.Text = username;
            URLTextBox.Text = url;
        }

        // Save the User Info with the frame to the database.
        private void SaveButton_Click(object sender, EventArgs e)
        {
            if (!String.IsNullOrEmpty(UserNameTextBox.Text) && !String.IsNullOrEmpty(URLTextBox.Text))
            {
                FaceBot.SaveFaceWithUserInfo(UserNameTextBox.Text, URLTextBox.Text, FaceBot.File);
                Close();
            }
            else
                MessageBox.Show("Invalid or Empty input, please try again.");
        }
    }
}
