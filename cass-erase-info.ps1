for ( $i=0; $i -lt $args.count; $i++ )
{
    $f = "$($args[$i])"
    $info = rar p -inul "$f" Patient_info.bin
    $x = $info.split(',')
    $x[0] = "NA"
    $x[1] = "19000102"
    $x[0] = "NA"
    $x -join ',' | rar u -inul -si"Patient_info.bin" "$f" 
    "" | rar u -inul -si"Patient_info_new2.bin" "$f" 
}
