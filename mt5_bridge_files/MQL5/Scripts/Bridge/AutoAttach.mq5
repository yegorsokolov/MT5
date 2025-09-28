#property script_show_inputs
#property strict

input string TargetSymbol="EURUSD";
input ENUM_TIMEFRAMES TargetPeriod=PERIOD_M1;
input string ExpertPath="Experts\\Bridge\\BridgeEA";

int OnStart()
{
  long chart_id = ChartFirst();
  if(chart_id==-1)
  {
    chart_id = ChartOpen(TargetSymbol, TargetPeriod);
  }
  if(chart_id==-1)
    return -1;
  if(!ChartSetSymbolPeriod(chart_id, TargetSymbol, TargetPeriod))
    Print("Failed to set symbol/period");
  if(!ChartApplyTemplate(chart_id, "BridgeEA"))
    Print("Template apply failed");
  return 0;
}
