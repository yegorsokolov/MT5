#property copyright "File bridge EA"
#property link      "https://github.com"
#property version   "1.00"
#property strict

input string BridgeFolder="bridge";
input string CommandFile="command.json";
input string ResponseFile="response.json";
input string TickFile="tick.json";
input int IdleSleepMs=250;

string basePath;
string commandsPath;
string responsePath;
string tickPath;
string processedId;
datetime lastPing=0;

string ReadFile(const string name) {
  int handle = FileOpen(name, FILE_READ|FILE_TXT|FILE_ANSI);
  if(handle==INVALID_HANDLE)
    return "";
  string content = FileReadString(handle);
  FileClose(handle);
  return content;
}

void WriteFile(const string name,const string content) {
  int handle = FileOpen(name, FILE_WRITE|FILE_TXT|FILE_ANSI);
  if(handle==INVALID_HANDLE)
    return;
  FileWriteString(handle, content);
  FileClose(handle);
}

void EnsureBridge() {
  basePath = BridgeFolder;
  if(StringLen(basePath)==0)
    basePath = "bridge";
  commandsPath = basePath + "\\" + CommandFile;
  responsePath = basePath + "\\" + ResponseFile;
  tickPath = basePath + "\\" + TickFile;
  if(!FileIsExist(basePath))
    FileCreateDirectory(basePath);
}

int OnInit() {
  EnsureBridge();
  processedId = "";
  lastPing = TimeCurrent();
  Print("BridgeEA initialized at ", TimeToString(lastPing, TIME_DATE|TIME_SECONDS));
  EventSetTimer(1);
  return INIT_SUCCEEDED;
}

void PublishTick() {
  MqlTick tick;
  if(SymbolInfoTick(_Symbol, tick)) {
    string payload = "{\"symbol\":\"" + _Symbol + "\",\"time\":" + (string)tick.time +
                     ",\"bid\":" + DoubleToString(tick.bid, _Digits) +
                     ",\"ask\":" + DoubleToString(tick.ask, _Digits) +
                     ",\"last\":" + DoubleToString(tick.last, _Digits) + "}";
    WriteFile(tickPath, payload);
  }
}

void ProcessCommands() {
  if(!FileIsExist(commandsPath))
    return;
  string content = ReadFile(commandsPath);
  if(StringLen(content)==0)
    return;
  string id = "";
  string cmd = content;
  int delim = StringFind(content, ":");
  if(delim>0) {
    id = StringSubstr(content, 0, delim);
    cmd = StringSubstr(content, delim+1);
  }
  if(id==processedId && id!="")
    return;
  string result = "{";
  if(StringLen(id)>0)
    result += "\"id\":\"" + id + "\",";
  result += "\"status\":\"";
  if(StringCompare(cmd, "ping")==0) {
    result += "pong\"";
  } else {
    result += "unknown\"";
  }
  MqlTick tick;
  if(SymbolInfoTick(_Symbol, tick)) {
    result += ",\"symbol\":\"" + _Symbol + "\",\"bid\":" + DoubleToString(tick.bid,_Digits) +
              ",\"ask\":" + DoubleToString(tick.ask,_Digits) +
              ",\"time\":" + (string)tick.time;
  }
  result += "}";
  WriteFile(responsePath, result);
  processedId = id;
}

void OnTick() {
  PublishTick();
  ProcessCommands();
}

void OnTimer() {
  PublishTick();
  ProcessCommands();
}

void OnDeinit(const int reason) {
  EventKillTimer();
  Print("BridgeEA deinitialized");
}
