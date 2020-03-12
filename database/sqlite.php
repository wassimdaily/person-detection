<?php
$db = new SQLite3('Images.db');
$sql = "SELECT * FROM $my_table";  
$result = $db->query($sql);
while ($row = $result->fetchArray(SQLITE3_ASSOC)){
echo "<tr>";
            echo "<td>" . $record['name'] . "</td>";
            echo "<td>" . $record['start_time'] . "</td>";
            echo "<td>" . $record['end_time'] . "</td>";
            echo "<td>" . $record['image'] . "</td>";
            echo "<td>" . $record['person'] . "</td>";
            echo "</tr>";
}
unset($db); 
