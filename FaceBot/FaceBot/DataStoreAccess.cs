using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SQLite;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FaceBot
{
    // Interface that defines all member methods that need to be impletented for a successful database operation.
    interface IDataStoreAccess
    {
        void Initialize();

        bool SaveFace(String username, String url, Byte[] faceImage);

        List<Face> GetFaces();

        List<Face> GetFaces(String username);

        String GetUserName(String id);
    }

    class DataStoreAccess : IDataStoreAccess
    {
        private SQLiteConnection SQLiteConnection;

        public DataStoreAccess(String databasePath)
        { SQLiteConnection = new SQLiteConnection(String.Format("Data Source={0};Version=3;", databasePath)); }

        /// <summary>
        /// Initializes the database with two tables: faces, users.
        /// </summary>
        public void Initialize()
        {
            try
            {
                SQLiteConnection.Open();
                var createQuery = "CREATE TABLE faces (" +
                    "Id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, " +
                    "Image BLOB NOT NULL," +
                    "DateCaptured DATETIME," +
                    "UserId TEXT NOT NULL);";
                var cmd = new SQLiteCommand(createQuery, SQLiteConnection);
                var result = cmd.ExecuteNonQuery();

                createQuery = "CREATE TABLE users (" +
                    "Id TEXT PRIMARY KEY NOT NULL, " +
                    "UserName TEXT NOT NULL," +   
                    "URL TEXT NOT NULL);";
                cmd = new SQLiteCommand(createQuery, SQLiteConnection);
                result = cmd.ExecuteNonQuery();
            }
            catch (Exception) { return; }
            finally { SQLiteConnection.Close(); }
        }

        /// <summary>
        ///  Save the user's face with additional user info. 
        /// </summary>
        /// <param name="username">Username of the User to be saved.</param>
        /// <param name="url">URL that is saved along with the User to launch upon successful recognition.</param>
        /// <param name="faceImage">Face shot of the user.</param>
        /// <returns></returns>
        public bool SaveFace(String username, String url, Byte[] faceImage)
        {
            try
            {
                bool IsNewUser = false;
                String exisitingUserId = GetUserId(username);
                if (String.IsNullOrEmpty(exisitingUserId))
                {
                    IsNewUser = true;
                    exisitingUserId = Guid.NewGuid().ToString();
                }
                SQLiteConnection.Open();
                var insertQuery = "INSERT INTO faces (Image, DateCaptured, UserId) VALUES (@faceImage, @date, @userId)";
                var cmd = new SQLiteCommand(insertQuery, SQLiteConnection);
                cmd.Parameters.Add("faceImage", DbType.Binary, faceImage.Length).Value = faceImage;
                cmd.Parameters.AddWithValue("date", DateTime.Now);
                cmd.Parameters.AddWithValue("userId", exisitingUserId);
                var result = cmd.ExecuteNonQuery();

                if(IsNewUser)
                {
                    insertQuery = "INSERT INTO users (Id, UserName, URL) VALUES (@id, @username, @url)";
                    cmd = new SQLiteCommand(insertQuery, SQLiteConnection);
                    cmd.Parameters.AddWithValue("id", exisitingUserId);
                    cmd.Parameters.AddWithValue("username", username);
                    cmd.Parameters.AddWithValue("url", url);
                    result = cmd.ExecuteNonQuery();
                }
                return true;
            }
            catch (Exception ex)
            {
                return false;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public String GetUserId(String username)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT Id FROM users WHERE UserName=@username;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("username", username);
                var result = cmd.ExecuteReader();
                string userId = "";
                if (result.Read())
                    userId = (String)result["Id"];
                else
                    userId = String.Empty;

                result.Close();

                return userId;
            }
            catch
            { return String.Empty; }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public string GetUserNameByFaceId(int faceId)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT UserId FROM faces WHERE Id=@faceId;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("faceId", faceId);
                var result = cmd.ExecuteReader();
                var userId = String.Empty;
                if (result.Read())
                {
                    userId = (String)result["UserId"];
                    result.Close();
                    SQLiteConnection.Close();
                    return GetUserName(userId);
                }
                else
                    return String.Empty;
            }
            catch
            {
                return String.Empty;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public User GetUserByFaceId(int faceId)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT UserId FROM faces WHERE Id=@faceId;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("faceId", faceId);
                var result = cmd.ExecuteReader();
                var userId = String.Empty;
                if (result.Read())
                {
                    userId = (String)result["UserId"];
                    result.Close();
                    SQLiteConnection.Close();
                    return GetUserById(userId);
                }
                else
                    return null;
            }
            catch
            {
                return null;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public String GetUserName(String id)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT UserName FROM users WHERE Id=@id;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("id", id);
                var result = cmd.ExecuteReader();
                string username = "";
                if (result.Read())
                    username = result.GetValue(0).ToString();
                else
                    username = String.Empty;

                result.Close();
                return username;
            }
            catch
            {
                return String.Empty;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public User GetUserById(String id)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT * FROM users WHERE Id=@id;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("id", id);
                var result = cmd.ExecuteReader();
                User user = null;
                if (result.Read())
                {
                    user = new User
                    {
                        UserName = result.GetValue(1).ToString(),
                        URL = result.GetValue(2).ToString()
                    };
                }

                result.Close();
                return user;
            }
            catch
            {
                return null;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public List<Face> GetFaces()
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT * FROM faces;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                var result = cmd.ExecuteReader();
                List<Face> faces = new List<Face>();
                while (result.Read())
                {
                    faces.Add(new Face
                    {
                        Id = Convert.ToInt32(result["Id"].ToString()),
                        DateCaptured = result["DateCaptured"] != DBNull.Value ? DateTime.Parse(result["DateCaptured"].ToString()) : new DateTime(),
                        UserId = (String)result["UserId"],
                        Image = (Byte[])result["Image"]
                    });
                }
                result.Close();
                return faces;
            }
            catch(Exception)
            {
                return null;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }

        public List<Face> GetFaces(String id)
        {
            try
            {
                SQLiteConnection.Open();
                var selectQuery = "SELECT * FROM faces WHERE UserId=@id;";
                var cmd = new SQLiteCommand(selectQuery, SQLiteConnection);
                cmd.Parameters.AddWithValue("id", id);
                var result = cmd.ExecuteReader();
                List<Face> faces = new List<Face>();
                while(result.Read())
                {
                    faces.Add(new Face
                    {
                        Id = Convert.ToInt32(result["Id"].ToString()),
                        DateCaptured = result["DateCaptured"] != DBNull.Value ? DateTime.Parse(result["DateCaptured"].ToString()) : new DateTime(),
                        UserId = (String)result["UserId"],
                        Image = (Byte[])result["Image"]
                    });
                }
                result.Close();
                return faces;
            }
            catch
            {
                return null;
            }
            finally
            {
                SQLiteConnection.Close();
            }
        }
    }
}
