#property script_show_inputs
#property strict

input long   ExpectedLogin   = 0;
input string ExpectedServer  = "";
input string SymbolToPing    = "EURUSD";
input bool   RequestMarketTick = true;

void PrintDivider()
{
   Print("--------------------------------------------------------------");
}

string FormatBool(const bool value)
{
   return value ? "YES" : "NO";
}

void ReportTerminalStatus()
{
   PrintDivider();
   PrintFormat("Terminal connected: %s", FormatBool(TerminalInfoInteger(TERMINAL_CONNECTED)));
   PrintFormat("Terminal build: %d", (int)TerminalInfoInteger(TERMINAL_BUILD));
   PrintFormat("Data center: %s", TerminalInfoString(TERMINAL_DATACENTER));
   PrintFormat("Server: %s", AccountInfoString(ACCOUNT_SERVER));
   PrintDivider();
}

void ReportAccountStatus()
{
   long login = AccountInfoInteger(ACCOUNT_LOGIN);
   string server = AccountInfoString(ACCOUNT_SERVER);
   ENUM_ACCOUNT_TRADE_MODE mode = (ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE);

   PrintFormat("Account: %I64d (%s)", login, AccountInfoString(ACCOUNT_NAME));
   PrintFormat("Leverage: 1:%d", (int)AccountInfoInteger(ACCOUNT_LEVERAGE));
   PrintFormat("Trade mode: %s", EnumToString(mode));

   if(ExpectedLogin > 0 && login != ExpectedLogin)
      PrintFormat("[WARNING] Connected login differs from ExpectedLogin input (%I64d)", ExpectedLogin);

   if(StringLen(ExpectedServer) > 0 && server != ExpectedServer)
      PrintFormat("[WARNING] Connected server differs from ExpectedServer input (%s)", ExpectedServer);
}

void ReportTimeSynchronisation()
{
   datetime server_time = TimeTradeServer();
   datetime local_time = TimeCurrent();
   PrintFormat("Server time: %s", TimeToString(server_time, TIME_DATE|TIME_SECONDS));
   PrintFormat("Local time:  %s", TimeToString(local_time, TIME_DATE|TIME_SECONDS));
}

void ProbeMarketData()
{
   if(!RequestMarketTick)
      return;

   string symbol = SymbolToPing;
   if(StringLen(symbol) == 0)
      symbol = _Symbol;

   if(!SymbolSelect(symbol, true))
   {
      PrintFormat("[ERROR] Unable to subscribe to %s.", symbol);
      return;
   }

   MqlTick tick;
   if(SymbolInfoTick(symbol, tick))
   {
      PrintFormat("Latest tick for %s: bid=%.5f ask=%.5f time=%s", symbol, tick.bid, tick.ask, TimeToString(tick.time, TIME_DATE|TIME_SECONDS));
   }
   else
   {
      PrintFormat("[ERROR] SymbolInfoTick failed for %s (error %d)", symbol, GetLastError());
   }
}

void OnStart()
{
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      Print("[ERROR] Terminal is not connected. Open Navigator → Accounts and log in with your broker credentials.");
      return;
   }

   ReportTerminalStatus();
   ReportAccountStatus();
   ReportTimeSynchronisation();
   ProbeMarketData();

   PrintDivider();
   Print("Heartbeat complete. If warnings are shown above, resolve them before starting the Python bridge.");
   PrintDivider();
}
