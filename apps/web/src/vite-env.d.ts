/// <reference types="vite/client" />

declare module '*?worklet' {
  const workletCode: string;
  export default workletCode;
}

declare module '*?blobworker' {
  const workerCode: string;
  export default workerCode;
}
