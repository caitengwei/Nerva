import http from "k6/http";
import { check } from "k6";

const baseUrl = __ENV.BASE_URL || "http://127.0.0.1:8000";
const deadlineMs = __ENV.DEADLINE_MS || "1893456000000";
const pipeline = __ENV.PIPELINE || "text_generation";
const reqBin = __ENV.REQ_BIN || "./unary_req.bin";
const payload = open(reqBin, "b");

const vus = Number(__ENV.CONCURRENCY || 32);
const duration = __ENV.DURATION || "300s";

export const options = {
  vus,
  duration,
};

export default function () {
  const res = http.post(`${baseUrl}/rpc/${pipeline}`, payload, {
    headers: {
      "Content-Type": "application/x-nerva-rpc",
      Accept: "application/x-nerva-rpc",
      "x-nerva-stream": "0",
      "x-nerva-deadline-ms": deadlineMs,
    },
  });

  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}

