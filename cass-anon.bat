
for %%f in (%*) do (
    echo %%f
    set s=%rar p -inul %%f Patient_info.bin%
    echo %s%
    @REM echo NAME,00001122,NA,$(rar p -inul "$f" Patient_info.bin | cut -d ',' -f4-) | rar u -inul -siPatient_info.bin "$f"
    @REM echo '' | rar u -inul -siPatient_info_new2.bin "$f"
    @REM echo $f done
)