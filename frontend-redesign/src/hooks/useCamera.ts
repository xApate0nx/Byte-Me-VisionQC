import { useCallback, useEffect, useRef, useState } from "react";

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export function useCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(false);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>("");
  const [resolution, setResolution] = useState<{ width: number; height: number } | null>(null);
  const [torchAvailable, setTorchAvailable] = useState(false);
  const [torchOn, setTorchOn] = useState(false);

  // Enumerate available video input devices
  const refreshDevices = useCallback(async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices
        .filter((d) => d.kind === "videoinput")
        .map((d, index) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${index + 1}`,
        }));
      setDevices(videoInputs);
      if (videoInputs.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoInputs[0].deviceId);
      }
    } catch (e) {
      console.warn("Could not enumerate camera devices:", e);
    }
  }, [selectedDeviceId]);

  const stop = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setActive(false);
    setResolution(null);
    setTorchOn(false);
  }, []);

  const start = useCallback(async (deviceIdOverride?: string) => {
    try {
      stop();
      setError(null);
      const targetDevice = deviceIdOverride || selectedDeviceId;
      const constraints: MediaStreamConstraints = {
        video: targetDevice
          ? { deviceId: { exact: targetDevice }, width: { ideal: 1280 }, height: { ideal: 960 } }
          : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setActive(true);

      const track = stream.getVideoTracks()[0];
      if (track) {
        const settings = track.getSettings();
        if (settings.width && settings.height) {
          setResolution({ width: settings.width, height: settings.height });
        }
        // Check torch capability
        const capabilities = track.getCapabilities?.() as { torch?: boolean } | undefined;
        setTorchAvailable(!!capabilities?.torch);
      }

      await refreshDevices();
    } catch (err) {
      const message = err instanceof Error ? err.message : "Camera access was denied or device unavailable.";
      setError(message);
      setActive(false);
    }
  }, [selectedDeviceId, stop, refreshDevices]);

  const switchDevice = useCallback((deviceId: string) => {
    setSelectedDeviceId(deviceId);
    if (active) {
      start(deviceId);
    }
  }, [active, start]);

  const toggleTorch = useCallback(async () => {
    if (!videoRef.current?.srcObject || !torchAvailable) return;
    const stream = videoRef.current.srcObject as MediaStream;
    const track = stream.getVideoTracks()[0];
    if (track) {
      try {
        const nextState = !torchOn;
        await (track as any).applyConstraints({
          advanced: [{ torch: nextState }],
        });
        setTorchOn(nextState);
      } catch (e) {
        console.warn("Torch control not supported:", e);
      }
    }
  }, [torchAvailable, torchOn]);

  const capture = useCallback(async (): Promise<Blob> => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      throw new Error("Camera feed is not ready. Please start the camera and allow video preview.");
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not create canvas context for image capture.");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error("Failed to encode camera snapshot into JPEG format."));
          }
        },
        "image/jpeg",
        0.95
      );
    });
  }, []);

  useEffect(() => {
    refreshDevices();
    return () => {
      stop();
    };
  }, [refreshDevices, stop]);

  return {
    videoRef,
    error,
    active,
    devices,
    selectedDeviceId,
    resolution,
    torchAvailable,
    torchOn,
    start,
    stop,
    switchDevice,
    toggleTorch,
    capture,
  };
}
