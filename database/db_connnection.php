 <?php
$servername = "localhost";
$username = "root";
$password = "Hydro-1";

// Create connection
$conn = mysqli_connect($servername, $username, $password);

// Check connection
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}
echo "Connected successfully";

//Setup our query 
$query = "SELECT * FROM my_table";     

//Run the Query 
$result = mysql_query($query);

//If the query returned results, loop through 
// each result 

if($result) 
    {   while($row = mysql_fetch_array($result))   
        {     $name = $row["name"];     
              echo "Name: " . $name; 
   } }   

?>
