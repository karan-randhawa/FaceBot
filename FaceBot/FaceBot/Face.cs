using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FaceBot
{
    // Face class to store face data about a User in the database.
    internal class Face
    {
        public Byte[] Image { get; set; }
        public Int32 Id { get; set; }
        public DateTime? DateCaptured { get; set; }
        public String UserId { get; set; }
    }

    // User class to store user information along with the Face data in the database.
    public class User
    {
        public String Id { get; set; }
        public String UserName { get; set; }
        public String URL { get; set; }
        public double Distance { get; set; }
        public Rectangle Face { get; set; }

        public User()
        {
            Id = string.Empty;
            UserName = string.Empty;
            URL = string.Empty;
            Distance = 0;
            Face = new Rectangle();
        }
    }
}
