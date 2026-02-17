interface FeatureCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
}

export default function FeatureCard({ icon: Icon, title, description }: FeatureCardProps) {
  return (
    <div className="rounded-xl bg-white/5 p-6 ring-1 ring-white/10 backdrop-blur-sm">
      <Icon className="h-8 w-8 text-indigo-400" />
      <h3 className="mt-4 text-lg font-semibold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-indigo-200">{description}</p>
    </div>
  );
}
